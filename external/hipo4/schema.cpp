#include "schema.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <ranges>

namespace hipo {

namespace {

    constexpr std::string_view whitespace = " \t\n\r";

    [[nodiscard]] std::string_view trim(std::string_view s) noexcept {
        const auto b = s.find_first_not_of(whitespace);
        if (b == std::string_view::npos) return {};
        const auto e = s.find_last_not_of(whitespace);
        return s.substr(b, e - b + 1);
    }

    /// Returns the content of the n-th `{...}` group of `s`, or nullopt.
    [[nodiscard]] std::optional<std::string_view> brace_group(std::string_view s, int n) noexcept {
        std::size_t pos = 0;
        for (int i = 0; i <= n; ++i) {
            const auto open = s.find('{', pos);
            if (open == std::string_view::npos) return std::nullopt;
            const auto close = s.find('}', open + 1);
            if (close == std::string_view::npos) return std::nullopt;
            if (i == n) return s.substr(open + 1, close - open - 1);
            pos = close + 1;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dtype> dtype_from_letter(std::string_view t) noexcept {
        if (t.size() != 1) return std::nullopt;
        switch (t.front()) {
            case 'B': return dtype::i8;
            case 'S': return dtype::i16;
            case 'I': return dtype::i32;
            case 'F': return dtype::f32;
            case 'D': return dtype::f64;
            case 'L': return dtype::i64;
            default:  return std::nullopt;
        }
    }

    [[nodiscard]] constexpr char letter_of(dtype t) noexcept {
        switch (t) {
            case dtype::i8:  return 'B';
            case dtype::i16: return 'S';
            case dtype::i32: return 'I';
            case dtype::f32: return 'F';
            case dtype::f64: return 'D';
            case dtype::i64: return 'L';
        }
        return '?';
    }

    template <class Int>
    [[nodiscard]] std::optional<Int> parse_int(std::string_view s) noexcept {
        s = trim(s);
        Int value{};
        const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc{} || ptr != s.data() + s.size()) return std::nullopt;
        return value;
    }

} // namespace

std::optional<std::uint32_t> bank_schema::index_of(std::string_view name) const noexcept {
    const auto it = std::ranges::lower_bound(
        by_name_, name, std::less{}, [&](std::uint32_t i) -> std::string_view { return cols_[i].name; });
    if (it == by_name_.end() || cols_[*it].name != name) return std::nullopt;
    return *it;
}

result<bank_schema> bank_schema::parse(std::string_view entry) {
    const auto fail = [&](std::string_view why) {
        return std::unexpected(error{errc::no_dictionary,
                                     std::format("bank_schema::parse: {} in \"{}\"", why, entry)});
    };

    const auto head = brace_group(entry, 0);
    const auto body = brace_group(entry, 1);
    if (!head || !body) return fail("expected {name/group/item}{columns}");

    bank_schema schema;
    {
        auto parts = std::views::split(*head, '/');
        auto it    = parts.begin();
        const auto next = [&]() -> std::optional<std::string_view> {
            if (it == parts.end()) return std::nullopt;
            const auto sv = std::string_view(*it);
            ++it;
            return sv;
        };
        const auto name  = next();
        const auto group = next();
        const auto item  = next();
        if (!name || !group || !item) return fail("malformed header");
        schema.name_ = trim(*name);
        const auto g = parse_int<std::uint16_t>(*group);
        const auto i = parse_int<std::uint16_t>(*item);
        if (schema.name_.empty() || !g || !i) return fail("malformed header");
        schema.group_ = *g;
        schema.item_  = *i;
    }

    std::uint32_t offset = 0;
    for (const auto col : std::views::split(*body, ',')) {
        const auto col_sv = trim(std::string_view(col));
        if (col_sv.empty()) continue;
        const auto slash = col_sv.find('/');
        if (slash == std::string_view::npos) return fail("column without type");
        const auto name = trim(col_sv.substr(0, slash));
        const auto type = dtype_from_letter(trim(col_sv.substr(slash + 1)));
        if (name.empty() || !type) return fail("malformed column");
        schema.cols_.push_back({std::string(name), offset, *type});
        offset += static_cast<std::uint32_t>(size_of(*type));
    }
    if (schema.cols_.empty()) return fail("no columns");
    schema.row_size_ = offset;

    schema.by_name_.resize(schema.cols_.size());
    for (std::uint32_t i = 0; i < schema.by_name_.size(); ++i) schema.by_name_[i] = i;
    std::ranges::sort(schema.by_name_, std::less{},
                      [&](std::uint32_t i) -> std::string_view { return schema.cols_[i].name; });
    return schema;
}

result<bank_schema> bank_schema::make(std::string_view name, std::uint16_t group,
                                      std::uint16_t item,
                                      std::span<const column_decl> columns) {
    const auto fail = [&](std::string_view why) {
        return std::unexpected(
            error{errc::no_dictionary, std::format("bank_schema::make(\"{}\"): {}", name, why)});
    };
    const auto valid_name = [](std::string_view n) {
        return !n.empty() && n.find_first_of("/{},") == std::string_view::npos;
    };
    if (!valid_name(name)) return fail("invalid bank name");
    if (columns.empty()) return fail("no columns");

    bank_schema schema;
    schema.name_  = std::string(name);
    schema.group_ = group;
    schema.item_  = item;
    std::uint32_t offset = 0;
    for (const auto& c : columns) {
        if (!valid_name(c.name) || size_of(c.type) == 0)
            return fail(std::format("invalid column \"{}\"", c.name));
        schema.cols_.push_back({std::string(c.name), offset, c.type});
        offset += static_cast<std::uint32_t>(size_of(c.type));
    }
    schema.row_size_ = offset;

    schema.by_name_.resize(schema.cols_.size());
    for (std::uint32_t i = 0; i < schema.by_name_.size(); ++i) schema.by_name_[i] = i;
    std::ranges::sort(schema.by_name_, std::less{},
                      [&](std::uint32_t i) -> std::string_view { return schema.cols_[i].name; });
    for (std::size_t i = 1; i < schema.by_name_.size(); ++i)
        if (schema.cols_[schema.by_name_[i - 1]].name == schema.cols_[schema.by_name_[i]].name)
            return fail(std::format("duplicate column \"{}\"",
                                    schema.cols_[schema.by_name_[i]].name));
    return schema;
}

std::string bank_schema::schema_string() const {
    std::string out = std::format("{{{}/{}/{}}}{{", name_, group_, item_);
    for (std::size_t i = 0; i < cols_.size(); ++i) {
        if (i != 0) out += ',';
        out += cols_[i].name;
        out += '/';
        out += letter_of(cols_[i].type);
    }
    out += '}';
    return out;
}

std::string bank_schema::schema_string_json() const {
    // byte-for-byte the legacy getSchemaStringJson layout (dictionary
    // records produced by old and new writers must match)
    std::string out = std::format("{{ \"name\": \"{}\", \"group\": {}, \"item\": {}, "
                                  "\"info\": \" \",\"entries\": [ ",
                                  name_, group_, item_);
    for (std::size_t i = 0; i < cols_.size(); ++i) {
        if (i != 0) out += ',';
        out += std::format("{{\"name\":\"{}\", \"type\":\"{}\", \"info\":\" \"}}",
                           cols_[i].name, letter_of(cols_[i].type));
    }
    out += "] }";
    return out;
}

} // namespace hipo
