// hipo-check: forensic reader / validator / re-indexer for (possibly
// malformed) HIPO files.
//
// Stock `hipo-utils -dump` reports "0 records / 0 events" when a file's TRAILER
// INDEX is missing -- it reads only through that footer record (the one that
// lists every data record's byte position). A reconstruction job killed before
// close(), or a truncated/partial copy, loses the trailer while the event
// records themselves stay intact in the file body. That is exactly the "file
// does not appear to have an index ... use hipoutils -doctor" warning -- and on
// farms where `hipo-utils` has no `-doctor` command, every other hipo-utils
// subcommand (-merge, -filter, ...) is just as stuck, because they all go
// through the same Java reader that needs the index.
//
// This project's reader (external/hipo4) does NOT depend on the trailer: when
// it is absent or unreadable, file::init() falls back to a SEQUENTIAL SCAN of
// the record-header chain (build_index_sequential) and serves the readable
// prefix, flagging file::truncated() if the file ends mid-record.
//
// hipo-check uses that path two ways:
//   * default (read-only): report the recovered record/event counts, whether
//     the file is truncated, the bank dictionary, and a content tally
//     (RUN::config field scale + REC::Particle rows) proving real physics
//     decodes.
//   * --rewrite OUT: re-emit a properly-indexed copy. Every event is copied
//     VERBATIM (file_writer's skim path -- byte-identical bank payloads, tags
//     preserved) under a freshly written dictionary + trailer index, then the
//     output is reopened and its event count is verified against the input.
//     This is the `-doctor` the farm's hipo-utils is missing.
//
// Exit status is 0 on success (all inputs readable / the rewrite verified).
//
// Usage:
//   hipo-check <input.hipo>... [--list-banks] [--dump N] [--max-events N]
//   hipo-check --rewrite <out.hipo> <in.hipo>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "argparse/argparse.hpp"
#include "hipo4/hipo.hpp"

namespace {

// "1234567" -> "1,234,567" (locale-free: the project plots avoid locale/LaTeX).
auto commas(std::uint64_t v) -> std::string {
    std::string s = std::to_string(v), out;
    int n = 0;
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (n && n % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++n;
    }
    return {out.rbegin(), out.rend()};
}

struct cli_args {
    std::vector<std::string> inputs;
    std::string rewrite;       // when set: write a re-indexed copy here and verify
    bool list_banks = false;
    long long dump = 0;        // print the raw structures of the first N events
    long long max_events = 0;  // content tally over at most N events (0 = all)
};

auto build_parser() -> argparse::parser {
    argparse::parser p(
        "hipo-check",
        "Validate, recover-read, and (with --rewrite) re-index a possibly index-less / "
        "truncated HIPO file -- the case where stock `hipo-utils -dump` shows 0 records.");
    p.add_argument("inputs")
        .nargs(argparse::nargs::one_or_more)
        .metavar("input")
        .help(".hipo file(s) to check");
    p.add_argument("--rewrite", "--doctor")
        .default_value("")
        .metavar("OUT")
        .help("write a re-indexed copy to OUT (events copied verbatim) and verify it; "
              "takes exactly one input");
    p.add_argument("--list-banks").flag().help("print every bank schema (name, group/item, columns)");
    p.add_argument("--dump")
        .default_value(static_cast<long long>(0))
        .help("print the raw structures (group/item/type/size) of the first N events");
    p.add_argument("--max-events")
        .default_value(static_cast<long long>(0))
        .help("run the content tally over at most N events (0 = all)");
    return p;
}

auto read_args(const argparse::parser& p) -> cli_args {
    cli_args a;
    a.inputs = p.get<std::vector<std::string>>("inputs");
    a.rewrite = p.get<std::string>("--rewrite");
    a.list_banks = p.get<bool>("--list-banks");
    a.dump = p.get<long long>("--dump");
    a.max_events = p.get<long long>("--max-events");
    return a;
}

// Cheap header block, straight from the index -- no event iteration.
auto print_header(const hipo::file& f, const std::string& path) -> void {
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    fmt::print("file: {}\n", path);
    if (!ec) fmt::print("  size           : {} bytes\n", commas(bytes));
    fmt::print("  schemas (banks): {}\n", f.schemas().size());
    fmt::print("  records        : {}\n", commas(f.record_count()));
    fmt::print("  events         : {}\n", commas(f.event_count()));
    fmt::print("  truncated      : {}\n",
               f.truncated() ? "YES -- ends mid-record; partial tail dropped (writer was killed?)"
                             : "no");
}

// One distinct RUN::config seen while scanning, grouped by run number. A pass
// cooked at the wrong run shows up here as run 0 (or a run disagreeing with the
// file name); the per-run event counts tell whether a stray value is just the
// first scaler/config event or the whole file -- the difference between "cooked
// at run 0" and "harmless leading event".
struct run_cfg {
    int run = 0;
    double torus = 0;
    double solenoid = 0;
    long long count = 0;
    long long first_ev = 0;
};

// Fold one RUN::config row into the per-run tally (defensive against a
// dictionary missing a column). Groups by run number; keeps the field scales
// and event index of the first row seen for that run.
auto tally_run_config(std::vector<run_cfg>& cfgs, const hipo::bank_view& c, long long ev) -> void {
    int run = 0;
    double tor = 0, sol = 0;
    try {
        run = c.get<int>("run", 0);
        tor = c.get<double>("torus", 0);
        sol = c.get<double>("solenoid", 0);
    } catch (const std::exception&) { /* leave zeros */ }
    for (auto& rc : cfgs)
        if (rc.run == run) {
            ++rc.count;
            return;
        }
    cfgs.push_back({run, tor, sol, 1, ev});
}

auto check_one(const std::string& path, const cli_args& args) -> bool {
    auto fr = hipo::file::open(path);
    if (!fr) {
        fmt::print("file: {}\n  ERROR: cannot open: {}\n\n", path, fr.error().message);
        return false;
    }
    hipo::file& f = *fr;
    print_header(f, path);

    if (args.list_banks) {
        fmt::print("  banks:\n");
        for (const auto& s : f.schemas())
            fmt::print("    {:<28} (g {:>5}, i {:>3})  {:>3} cols, row {:>5} B\n", s.name(),
                       s.group(), s.item(), s.columns().size(), s.row_size());
    }

    if (f.event_count() == 0) {
        fmt::print("  VERDICT: NO readable events -- the body holds no decodable HIPO records.\n\n");
        return false;
    }

    // Single pass over events: capture the first RUN::config (field scale),
    // tally REC::Particle, and optionally dump the first N events' structures.
    auto cfg = f.bank("RUN::config");
    auto part = f.bank("REC::Particle");
    std::vector<run_cfg> cfgs;  // distinct RUN::config run values seen, with counts
    long long n_seen = 0, n_with_particle = 0, total_particles = 0;
    const long long cap = args.max_events > 0 ? args.max_events : -1;

    try {
        for (hipo::event_view ev : f) {
            if (cap >= 0 && n_seen >= cap) break;

            if (args.dump > 0 && n_seen < args.dump) {
                fmt::print("  event {} structures:\n", n_seen);
                for (auto s : ev.structures())
                    fmt::print("    group {:>5}  item {:>3}  type {:>2}  {:>7} B\n",
                               static_cast<unsigned>(s.group), static_cast<unsigned>(s.item),
                               static_cast<unsigned>(s.type), s.payload.size());
            }

            if (cfg) {
                hipo::bank_view c = ev.get(*cfg);
                if (c.rows() > 0) tally_run_config(cfgs, c, n_seen);
            }
            if (part) {
                hipo::bank_view pv = ev.get(*part);
                if (pv.rows() > 0) {
                    ++n_with_particle;
                    total_particles += pv.rows();
                }
            }
            ++n_seen;
        }
    } catch (const std::exception& e) {
        // A decode error mid-iteration: report what we got rather than abort.
        fmt::print("  WARNING: stopped at event {} ({})\n", n_seen, e.what());
    }

    if (cfgs.empty()) {
        fmt::print("  RUN::config    : (none found in scanned events)\n");
    } else {
        std::sort(cfgs.begin(), cfgs.end(),
                  [](const run_cfg& a, const run_cfg& b) { return a.run < b.run; });
        fmt::print("  RUN::config    : {} distinct run value{} over {} scanned events:\n",
                   cfgs.size(), cfgs.size() == 1 ? "" : "s", commas(n_seen));
        for (const auto& rc : cfgs)
            fmt::print("      run {:<8} {:>12} events (first at #{}), torus {:g}, solenoid {:g}\n",
                       rc.run, commas(rc.count), commas(rc.first_ev), rc.torus, rc.solenoid);
    }
    fmt::print("  events scanned : {}{}\n", commas(n_seen),
               cap >= 0 && n_seen >= cap ? " (capped by --max-events)" : "");
    if (part)
        fmt::print("  REC::Particle  : present in {} events, {} particle rows total\n",
                   commas(n_with_particle), commas(total_particles));
    else
        fmt::print("  REC::Particle  : bank not in this file's dictionary\n");

    fmt::print("  VERDICT: OK -- {} events recovered{}.\n\n", commas(f.event_count()),
               f.truncated() ? " (file truncated; data past the last complete record is lost)" : "");
    return true;
}

// Re-emit a properly-indexed copy of `in` at `out`: recover-read the source,
// mirror its dictionary + configuration, copy every event verbatim, write a
// fresh trailer index, then reopen `out` and verify the event count.
auto do_rewrite(const std::string& in, const std::string& out) -> bool {
    std::error_code ec;
    if (std::filesystem::exists(out) && std::filesystem::equivalent(in, out, ec)) {
        fmt::print(stderr, "error: --rewrite output must differ from the input\n");
        return false;
    }
    if (const auto parent = std::filesystem::path(out).parent_path(); !parent.empty())
        std::filesystem::create_directories(parent, ec);

    auto fr = hipo::file::open(in);
    if (!fr) {
        fmt::print(stderr, "error: cannot open '{}': {}\n", in, fr.error().message);
        return false;
    }
    hipo::file& f = *fr;
    print_header(f, in);

    const std::uint64_t total = f.event_count();
    if (total == 0) {
        fmt::print(stderr, "error: no recoverable events in '{}'; nothing to rewrite\n", in);
        return false;
    }
    if (f.truncated())
        fmt::print("  note: input is truncated; the copy will hold the {} recoverable events.\n",
                   commas(total));

    auto wr = hipo::file_writer::open(out);
    if (!wr) {
        fmt::print(stderr, "error: cannot create '{}': {}\n", out, wr.error().message);
        return false;
    }
    hipo::file_writer& w = *wr;

    // Dictionary + file configuration must be registered before the first write.
    for (const auto& s : f.schemas())
        if (auto r = w.add_schema(s); !r) {
            fmt::print(stderr, "error: add_schema('{}') failed: {}\n", s.name(), r.error().message);
            return false;
        }
    for (const auto& [k, v] : f.user_config())
        if (auto r = w.config(k, v); !r) {
            fmt::print(stderr, "error: config('{}') failed: {}\n", k, r.error().message);
            return false;
        }

    fmt::print("rewriting {} events: {} -> {}\n", commas(total), in, out);
    std::uint64_t n = 0;
    try {
        for (hipo::event_view ev : f) {
            if (auto r = w.write(ev); !r) {
                fmt::print(stderr, "\nerror: write failed at event {}: {}\n", n, r.error().message);
                return false;
            }
            if ((++n & 0x3FFFu) == 0) {  // ~every 16k events
                fmt::print("\r  {} / {} events", commas(n), commas(total));
                std::fflush(stdout);
            }
        }
    } catch (const std::exception& e) {
        fmt::print(stderr, "\nerror: read failed at event {}: {}\n", n, e.what());
        return false;
    }
    if (auto r = w.close(); !r) {
        fmt::print(stderr, "\nerror: close failed: {}\n", r.error().message);
        return false;
    }
    fmt::print("\r  {} / {} events  done\n", commas(n), commas(total));

    // Verify: reopen the output -- it must now index via its trailer (not the
    // sequential fallback) and round-trip the event count.
    auto vr = hipo::file::open(out);
    if (!vr) {
        fmt::print(stderr, "error: cannot reopen '{}' to verify: {}\n", out, vr.error().message);
        return false;
    }
    const std::uint64_t got = vr->event_count();
    const bool ok = got == total && !vr->truncated();
    fmt::print("verify: {} -> {} events ({}), truncated {}\n", commas(total), commas(got),
               got == total ? "match" : "MISMATCH", vr->truncated() ? "YES" : "no");
    if (!ok) {
        fmt::print(stderr, "error: rewrite verification FAILED\n");
        return false;
    }
    std::error_code se;
    const auto bytes = std::filesystem::file_size(out, se);
    fmt::print("OK: wrote indexed copy {}{} -- any tool can read it now "
               "(e.g. `hipo-utils -info {}`).\n",
               out, se ? std::string{} : fmt::format(" ({} bytes)", commas(bytes)), out);
    return true;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    argparse::parser p = build_parser();
    try {
        p.parse(argc, argv);
    } catch (const argparse::help_requested&) {
        fmt::print("{}", p.help_text());
        return 0;
    } catch (const argparse::error& e) {
        fmt::print(stderr, "error: {}\n\n{}", e.what(), p.usage());
        return 2;
    }
    const cli_args args = read_args(p);

    if (!args.rewrite.empty()) {
        if (args.inputs.size() != 1) {
            fmt::print(stderr, "error: --rewrite takes exactly one input file (got {})\n",
                       args.inputs.size());
            return 2;
        }
        return do_rewrite(args.inputs.front(), args.rewrite) ? 0 : 1;
    }

    bool all_ok = true;
    for (const auto& in : args.inputs) all_ok &= check_one(in, args);

    fmt::print(
        "note: counts above come from THIS reader, which rebuilds a missing index by scanning\n"
        "      records; stock `hipo-utils -dump` needs the trailer index and reports 0 without it.\n"
        "      To write a properly-indexed copy any tool can read (this farm's hipo-utils has no\n"
        "      -doctor):  hipo-check --rewrite <out.hipo> <in.hipo>\n");
    return all_ok ? 0 : 1;
}
