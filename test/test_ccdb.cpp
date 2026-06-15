// Unit test for the CCDB SQLite reader (src/common/ccdb.hpp). Builds a tiny
// synthetic snapshot carrying the CCDB schema my reader queries, then checks the
// directory walk, run-range / variation selection, latest-assignment wins, and
// the value/table accessors.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>

#include <sqlite3.h>

#include "ccdb.hpp"

namespace {

int failures = 0;
auto check(bool c, const char* msg) -> void {
    if (!c) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++failures;
    }
}
auto approx(std::optional<double> v, double want) -> bool {
    return v.has_value() && std::abs(*v - want) < 1e-9;
}

auto build_snapshot(const std::string& path) -> bool {
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) return false;
    const char* sql = R"SQL(
        CREATE TABLE directories  (id INTEGER PRIMARY KEY, parentId INTEGER, name TEXT);
        CREATE TABLE typeTables   (id INTEGER PRIMARY KEY, name TEXT, directoryId INTEGER);
        CREATE TABLE columns      (id INTEGER PRIMARY KEY, typeId INTEGER, name TEXT);
        CREATE TABLE variations   (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE runRanges    (id INTEGER PRIMARY KEY, runMin INTEGER, runMax INTEGER);
        CREATE TABLE constantSets (id INTEGER PRIMARY KEY, vault TEXT, constantTypeId INTEGER);
        CREATE TABLE assignments  (id INTEGER PRIMARY KEY, constantSetId INTEGER,
                                   runRangeId INTEGER, variationId INTEGER);

        INSERT INTO directories VALUES (1, 0, 'geometry');
        INSERT INTO directories VALUES (2, 1, 'beam');
        INSERT INTO directories VALUES (3, 1, 'shifts');
        INSERT INTO typeTables VALUES (10, 'position', 2);
        INSERT INTO typeTables VALUES (11, 'solenoid', 3);
        INSERT INTO columns VALUES (100, 10, 'x');
        INSERT INTO columns VALUES (101, 10, 'y');
        INSERT INTO columns VALUES (102, 11, 'x');
        INSERT INTO columns VALUES (103, 11, 'y');
        INSERT INTO columns VALUES (104, 11, 'z');
        INSERT INTO variations VALUES (1, 'default');
        INSERT INTO runRanges  VALUES (1, 0, 100000);
        INSERT INTO runRanges  VALUES (2, 0, 100000);
        -- two beam assignments covering the run; the newer (higher id) must win
        INSERT INTO constantSets VALUES (1000, '9.9|9.9',       10);
        INSERT INTO constantSets VALUES (1001, '0.123|-0.045',  10);
        INSERT INTO constantSets VALUES (1002, '0|0|-3.0',      11);
        INSERT INTO assignments VALUES (1, 1000, 1, 1);
        INSERT INTO assignments VALUES (2, 1001, 2, 1);
        INSERT INTO assignments VALUES (3, 1002, 1, 1);
    )SQL";
    char* err = nullptr;
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
    if (err != nullptr) sqlite3_free(err);
    sqlite3_close(db);
    return rc == SQLITE_OK;
}

}  // namespace

auto main() -> int {
    namespace fs = std::filesystem;
    const std::string path = (fs::temp_directory_path() / "swim_ccdb_test.sqlite").string();
    std::error_code ec;
    fs::remove(path, ec);
    if (!build_snapshot(path)) {
        std::fprintf(stderr, "could not build the synthetic CCDB snapshot\n");
        return 1;
    }

    {
        ccdb::reader db(path);
        check(approx(db.value("/geometry/beam/position", 18614, "x"), 0.123), "beam x");
        check(approx(db.value("/geometry/beam/position", 18614, "y"), -0.045), "beam y");
        check(approx(db.value("/geometry/shifts/solenoid", 18614, "z"), -3.0), "solenoid z");
        // newest assignment (id 2 -> 0.123) wins over the older (id 1 -> 9.9)
        check(approx(db.value("/geometry/beam/position", 50, "x"), 0.123), "latest assignment wins");

        auto t = db.get("/geometry/shifts/solenoid", 18614);
        check(t.has_value() && t->columns.size() == 3 && t->n_rows == 1, "table shape");
        check(t.has_value() && approx(t->value(0, "z"), -3.0), "table value accessor");

        check(!db.value("/geometry/beam/position", 200000, "x").has_value(), "run out of range");
        check(!db.value("/geometry/nope", 18614, "x").has_value(), "unknown path");
        check(!db.value("/geometry/beam/position", 18614, "z").has_value(), "unknown column");
    }
    fs::remove(path, ec);

    // An incomplete snapshot (the CCDB data tables missing) must degrade to
    // nullopt rather than throw, so a partial export just falls back to defaults.
    {
        const std::string empty = (fs::temp_directory_path() / "swim_ccdb_empty.sqlite").string();
        fs::remove(empty, ec);
        sqlite3* db = nullptr;
        if (sqlite3_open(empty.c_str(), &db) == SQLITE_OK) {
            sqlite3_exec(db, "CREATE TABLE schemaVersions(id INTEGER);", nullptr, nullptr, nullptr);
            sqlite3_close(db);
            ccdb::reader r(empty);
            check(!r.value("/geometry/beam/position", 18614, "x").has_value(),
                  "incomplete snapshot -> nullopt (no throw)");
        }
        fs::remove(empty, ec);
    }

    if (failures == 0) std::printf("all ccdb tests passed\n");
    return failures;
}
