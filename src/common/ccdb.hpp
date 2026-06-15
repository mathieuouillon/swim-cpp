// Minimal read-only CCDB lookup over a SQLite snapshot (no libccdb dependency).
//
// CLAS12 stores per-run constants — beam position, magnet shifts, scales — in
// CCDB. Given a snapshot exported to SQLite (`ccdb dump ... -o ccdb.sqlite`, or
// the file shipped with a cook), this reads a constant table for a run:
//
//   ccdb::reader db("ccdb.sqlite");
//   auto beam = db.get("/geometry/beam/position", run);   // a whole table
//   double x = beam->value(0, "x").value_or(0.0);
//   auto z = db.value("/geometry/shifts/solenoid", run, "z");  // one cell
//
// It implements the common-case resolution only: the requested variation (default
// "default"), and the latest assignment (highest assignment id) whose run range
// contains the run. Variation inheritance, time-ordered selection and event
// ranges are NOT modelled — adequate for static geometry/beam constants.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sqlite3.h>

namespace ccdb {

/// A constant table fetched from CCDB: ordered column names plus the cell values
/// in row-major order (data[row * columns.size() + col]).
struct table {
    std::vector<std::string> columns;
    std::size_t n_rows = 0;
    std::vector<double> data;

    [[nodiscard]] auto col_index(std::string_view name) const -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < columns.size(); ++i)
            if (columns[i] == name) return i;
        return std::nullopt;
    }
    [[nodiscard]] auto value(std::size_t row, std::string_view col) const -> std::optional<double> {
        const auto c = col_index(col);
        if (!c || row >= n_rows) return std::nullopt;
        return data[row * columns.size() + *c];
    }
};

class reader {
  public:
    /// Open a CCDB SQLite snapshot read-only. Throws std::runtime_error if it
    /// cannot be opened.
    explicit reader(const std::string& sqlite_path) {
        const int rc = sqlite3_open_v2(sqlite_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr);
        if (rc != SQLITE_OK) {
            const std::string msg = db_ != nullptr ? sqlite3_errmsg(db_) : "out of memory";
            if (db_ != nullptr) sqlite3_close(db_);
            db_ = nullptr;
            throw std::runtime_error("ccdb: cannot open '" + sqlite_path + "': " + msg);
        }
    }
    ~reader() {
        if (db_ != nullptr) sqlite3_close(db_);
    }
    reader(const reader&) = delete;
    auto operator=(const reader&) -> reader& = delete;
    reader(reader&& o) noexcept : db_(o.db_) { o.db_ = nullptr; }

    /// Fetch the whole constant table at `path` for `run`, or nullopt when the
    /// path / run / variation has no assignment.
    [[nodiscard]] auto get(const std::string& path, int run,
                           const std::string& variation = "default") const -> std::optional<table> {
        const auto tt = type_table_id(path);
        if (!tt) return std::nullopt;
        const std::vector<std::string> cols = columns_of(*tt);
        if (cols.empty()) return std::nullopt;

        const auto vault = latest_vault(*tt, run, variation);
        if (!vault) return std::nullopt;

        table t;
        t.columns = cols;
        for (const std::string& tok : split(*vault, '|')) t.data.push_back(to_double(tok));
        t.n_rows = t.data.size() / cols.size();
        return t;
    }

    /// Convenience: one numeric cell (row 0) by column name.
    [[nodiscard]] auto value(const std::string& path, int run, const std::string& column,
                             const std::string& variation = "default") const
        -> std::optional<double> {
        const auto t = get(path, run, variation);
        if (!t) return std::nullopt;
        return t->value(0, column);
    }

  private:
    // ---- small SQLite helpers --------------------------------------------------
    // Returns nullptr on failure (e.g. a table missing from an incomplete
    // snapshot); callers treat that as "not found" and fall back to defaults.
    [[nodiscard]] auto prepare(const std::string& sql) const -> sqlite3_stmt* {
        sqlite3_stmt* st = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) return nullptr;
        return st;
    }

    /// Resolve "/a/b/name" to the typeTables.id of `name` under directory /a/b.
    [[nodiscard]] auto type_table_id(const std::string& path) const -> std::optional<long long> {
        const std::vector<std::string> seg = split(path, '/');  // empties dropped by split
        if (seg.empty()) return std::nullopt;

        // Walk the directory tree to the parent of the leaf. Top-level entries
        // hang off parentId 0 (some snapshots also carry an empty-named root).
        long long parent = 0;
        if (sqlite3_stmt* st = prepare("SELECT id FROM directories WHERE parentId=0 AND "
                                       "(name='' OR name IS NULL) LIMIT 1")) {
            if (sqlite3_step(st) == SQLITE_ROW) parent = sqlite3_column_int64(st, 0);
            sqlite3_finalize(st);
        }
        for (std::size_t i = 0; i + 1 < seg.size(); ++i) {
            sqlite3_stmt* st = prepare("SELECT id FROM directories WHERE name=?1 AND parentId=?2");
            if (st == nullptr) return std::nullopt;
            sqlite3_bind_text(st, 1, seg[i].c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int64(st, 2, parent);
            const bool ok = sqlite3_step(st) == SQLITE_ROW;
            if (ok) parent = sqlite3_column_int64(st, 0);
            sqlite3_finalize(st);
            if (!ok) return std::nullopt;
        }

        sqlite3_stmt* st = prepare("SELECT id FROM typeTables WHERE name=?1 AND directoryId=?2");
        if (st == nullptr) return std::nullopt;
        sqlite3_bind_text(st, 1, seg.back().c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(st, 2, parent);
        std::optional<long long> id;
        if (sqlite3_step(st) == SQLITE_ROW) id = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
        return id;
    }

    [[nodiscard]] auto columns_of(long long type_id) const -> std::vector<std::string> {
        std::vector<std::string> out;
        sqlite3_stmt* st = prepare("SELECT name FROM columns WHERE typeId=?1 ORDER BY id");
        if (st == nullptr) return out;
        sqlite3_bind_int64(st, 1, type_id);
        while (sqlite3_step(st) == SQLITE_ROW) {
            const auto* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            out.emplace_back(txt != nullptr ? txt : "");
        }
        sqlite3_finalize(st);
        return out;
    }

    /// The vault (raw `|`-delimited cell string) of the latest assignment whose
    /// run range contains `run`, in the given variation.
    [[nodiscard]] auto latest_vault(long long type_id, int run,
                                    const std::string& variation) const -> std::optional<std::string> {
        sqlite3_stmt* st = prepare(
            "SELECT cs.vault FROM assignments a "
            "JOIN constantSets cs ON cs.id = a.constantSetId "
            "JOIN runRanges rr   ON rr.id = a.runRangeId "
            "JOIN variations v    ON v.id  = a.variationId "
            "WHERE cs.constantTypeId = ?1 AND rr.runMin <= ?2 AND rr.runMax >= ?2 AND v.name = ?3 "
            "ORDER BY a.id DESC LIMIT 1");
        if (st == nullptr) return std::nullopt;
        sqlite3_bind_int64(st, 1, type_id);
        sqlite3_bind_int(st, 2, run);
        sqlite3_bind_text(st, 3, variation.c_str(), -1, SQLITE_TRANSIENT);
        std::optional<std::string> vault;
        if (sqlite3_step(st) == SQLITE_ROW) {
            const auto* txt = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            if (txt != nullptr) vault = txt;
        }
        sqlite3_finalize(st);
        return vault;
    }

    // ---- string helpers --------------------------------------------------------
    [[nodiscard]] static auto split(std::string_view s, char delim) -> std::vector<std::string> {
        std::vector<std::string> out;
        std::size_t start = 0;
        while (start <= s.size()) {
            const std::size_t end = s.find(delim, start);
            const std::string_view tok =
                s.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
            if (!tok.empty()) out.emplace_back(tok);
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
        return out;
    }
    [[nodiscard]] static auto to_double(const std::string& s) -> double {
        try {
            return std::stod(s);
        } catch (...) {
            return 0.0;  // non-numeric cell (e.g. a name) -> 0
        }
    }

    sqlite3* db_ = nullptr;
};

}  // namespace ccdb
