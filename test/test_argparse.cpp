// Unit tests for the header-only argparse library (external/argparse).
#include <cstdio>
#include <string>
#include <vector>

#include "argparse/argparse.hpp"

namespace {

int failures = 0;
auto check(bool c, const char* msg) -> void {
    if (!c) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}

// Parse `toks` (the args after argv[0]) through the (int, argv) entry point.
auto run(argparse::parser& p, std::vector<const char*> toks) -> void {
    std::vector<const char*> argv;
    argv.push_back("test");
    for (const char* t : toks) argv.push_back(t);
    p.parse(static_cast<int>(argv.size()), argv.data());
}

// A parser shaped like the ones the swim programs build.
auto make() -> argparse::parser {
    argparse::parser p("test", "a test parser");
    p.add_argument("inputs").nargs(argparse::nargs::one_or_more).help("inputs");
    p.add_argument("-o", "--output").default_value("out.root").help("output");
    p.add_argument("-j", "--jobs", "-t", "--threads").default_value(0).help("jobs");
    p.add_argument("-q", "--quiet").flag().help("quiet");
    p.add_argument("--scale").default_value(-1.0).help("scale");
    p.add_argument("--max").default_value(static_cast<long long>(-1)).help("max");
    p.add_argument("--region").default_value(3).choices({1, 3}).help("region");
    p.add_argument("--secret").hidden();
    return p;
}

template <class F>
auto throws_error(F&& f) -> bool {
    try {
        f();
    } catch (const argparse::error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

auto main() -> int {
    // Defaults applied when absent.
    {
        auto p = make();
        run(p, {"a.root"});
        check(p.get<std::string>("--output") == "out.root", "default string");
        check(p.get<int>("--jobs") == 0, "default int");
        check(p.get<double>("--scale") == -1.0, "default double");
        check(p.get<long long>("--max") == -1, "default long long");
        check(p.get<bool>("--quiet") == false, "default flag is false");
        check(p.get<int>("--region") == 3, "default choice");
        check(!p.is_used("--output"), "is_used false for unset option");
        const auto in = p.get<std::vector<std::string>>("inputs");
        check(in.size() == 1 && in[0] == "a.root", "single positional");
    }
    // Long / short / aliases / multiple positionals.
    {
        auto p = make();
        run(p, {"--output", "x.root", "-j", "4", "f1", "f2"});
        check(p.get<std::string>("--output") == "x.root", "long option value");
        check(p.get<int>("--jobs") == 4, "short alias -j");
        check(p.is_used("--jobs"), "is_used true when given");
        check(p.get<std::vector<std::string>>("inputs").size() == 2, "two positionals");
    }
    // `--opt=val` form and the --threads alias mapping to the same argument.
    {
        auto p = make();
        run(p, {"--threads", "8", "--output=y.root", "f"});
        check(p.get<int>("--jobs") == 8, "--threads alias -> jobs");
        check(p.get<std::string>("--output") == "y.root", "--output=val form");
    }
    // store-true flag.
    {
        auto p = make();
        run(p, {"-q", "f"});
        check(p.get<bool>("--quiet") == true, "flag set to true");
    }
    // Negative-number value is consumed (not read as an option).
    {
        auto p = make();
        run(p, {"--scale", "-3.5", "f"});
        check(p.get<double>("--scale") == -3.5, "negative value consumed");
    }
    // `--` ends option parsing.
    {
        auto p = make();
        run(p, {"--", "-weird-file"});
        const auto in = p.get<std::vector<std::string>>("inputs");
        check(in.size() == 1 && in[0] == "-weird-file", "-- end of options");
    }
    // Typed-conversion and choices errors.
    {
        auto p = make();
        check(throws_error([&] { run(p, {"--jobs", "abc", "f"}); }), "bad int errors");
    }
    {
        auto p = make();
        check(throws_error([&] { run(p, {"--scale", "xx", "f"}); }), "bad double errors");
    }
    {
        auto p = make();
        check(throws_error([&] { run(p, {"--region", "2", "f"}); }), "bad choice errors");
    }
    // Missing required positional / unknown option.
    {
        auto p = make();
        check(throws_error([&] { run(p, {"-q"}); }), "missing positional errors");
    }
    {
        auto p = make();
        check(throws_error([&] { run(p, {"--nope", "f"}); }), "unknown option errors");
    }
    // Missing value for an option that expects one.
    {
        auto p = make();
        check(throws_error([&] { run(p, {"f", "--output"}); }), "missing value errors");
    }
    // -h / --help signal via help_requested.
    {
        auto p = make();
        bool helped = false;
        try {
            run(p, {"--help"});
        } catch (const argparse::help_requested&) {
            helped = true;
        }
        check(helped, "--help -> help_requested");
    }
    {
        auto p = make();
        bool helped = false;
        try {
            run(p, {"-h"});
        } catch (const argparse::help_requested&) {
            helped = true;
        }
        check(helped, "-h -> help_requested");
    }
    // Help text excludes hidden args, includes visible ones + usage.
    {
        auto p = make();
        const std::string h = p.help_text();
        check(h.find("--secret") == std::string::npos, "hidden absent from help");
        check(h.find("--output") != std::string::npos, "visible option in help");
        check(h.find("usage:") != std::string::npos, "help contains usage line");
        check(p.usage().find("INPUTS") != std::string::npos, "usage lists the positional");
    }
    // A required *option* that is missing.
    {
        argparse::parser p("t");
        p.add_argument("--need").required();
        check(throws_error([&] { run(p, {}); }), "missing required option errors");
    }

    if (failures == 0) std::printf("all argparse tests passed\n");
    return failures;
}
