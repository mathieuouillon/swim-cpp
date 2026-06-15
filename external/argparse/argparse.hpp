// A small, self-contained C++23 command-line parser with a Python-argparse-
// flavored builder API. Header-only, zero dependencies (standard library only),
// so it drops into any project and builds anywhere a C++23 compiler does.
//
//   argparse::parser p("prog", "what it does");
//   p.add_argument("inputs").nargs(argparse::nargs::one_or_more).help("input files");
//   p.add_argument("-o", "--output").default_value(std::string("out.root")).help("output");
//   p.add_argument("-j", "--jobs", "--threads").default_value(0).help("0 = nproc");
//   p.add_argument("-q", "--quiet").flag().help("be quiet");
//   p.add_argument("--mode").default_value(std::string("fast")).choices({"fast", "slow"});
//   p.add_argument("--secret").hidden();             // parsed, omitted from --help
//
//   try {
//       p.parse(argc, argv);
//   } catch (const argparse::help_requested&) {
//       std::cout << p.help_text();      // -h / --help
//       return 0;
//   } catch (const argparse::error& e) {
//       std::cerr << "error: " << e.what() << "\n\n" << p.usage();
//       return 2;
//   }
//
//   int jobs = p.get<int>("--jobs");                       // any alias works
//   auto inputs = p.get<std::vector<std::string>>("inputs");
//   bool quiet = p.get<bool>("--quiet");
//
// Supported syntax: `--long`, `--long value`, `--long=value`, `-s`, `-s value`,
// and `--` (everything after is positional). Option values consume the next
// token even when it begins with `-` (so `--shift -3.0` works).
#pragma once

#include <algorithm>
#include <any>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace argparse {

/// Thrown on bad user input — the caller prints the message + usage and exits
/// non-zero. Derives std::runtime_error so a single `catch` covers it.
struct error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Thrown when `-h`/`--help` is seen — the caller prints `help_text()`, exit 0.
struct help_requested {};

/// How many values a (positional) argument consumes.
enum class nargs {
    one,           ///< exactly one value (the default)
    optional,      ///< zero or one; falls back to the default when absent
    zero_or_more,  ///< collect the rest into a std::vector<std::string>
    one_or_more,   ///< like zero_or_more but at least one is required
};

namespace detail {

enum class value_type { string_, int_, long_long_, double_, bool_ };

template <class T>
[[nodiscard]] constexpr auto type_tag() -> value_type {
    if constexpr (std::is_same_v<T, bool>) return value_type::bool_;
    else if constexpr (std::is_same_v<T, int>) return value_type::int_;
    else if constexpr (std::is_integral_v<T>) return value_type::long_long_;
    else if constexpr (std::is_floating_point_v<T>) return value_type::double_;
    else return value_type::string_;  // std::string, const char*, string_view, ...
}

/// Convert one token to a typed std::any per `vt`; throws `error` on a malformed
/// value or trailing garbage. `name` is used only for the message.
[[nodiscard]] inline auto convert(std::string_view tok, value_type vt, std::string_view name)
    -> std::any {
    const auto bad = [&](std::string_view what) -> std::any {
        throw error("argument " + std::string(name) + ": invalid " + std::string(what) +
                    " value: '" + std::string(tok) + "'");
    };
    switch (vt) {
        case value_type::string_:
            return std::any(std::string(tok));
        case value_type::int_: {
            int v{};
            const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
            if (ec != std::errc{} || ptr != tok.data() + tok.size()) return bad("int");
            return std::any(v);
        }
        case value_type::long_long_: {
            long long v{};
            const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
            if (ec != std::errc{} || ptr != tok.data() + tok.size()) return bad("integer");
            return std::any(v);
        }
        case value_type::double_: {
            // std::from_chars for floating point isn't available on every farm
            // toolchain yet; std::strtod is and is just as strict here.
            const std::string s(tok);
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (s.empty() || end != s.c_str() + s.size()) return bad("number");
            return std::any(v);
        }
        case value_type::bool_:
            return std::any(true);  // flags never convert a token
    }
    return std::any{};
}

[[nodiscard]] inline auto any_to_string(const std::any& a, value_type vt) -> std::string {
    switch (vt) {
        case value_type::string_: return std::any_cast<std::string>(a);
        case value_type::int_: return std::to_string(std::any_cast<int>(a));
        case value_type::long_long_: return std::to_string(std::any_cast<long long>(a));
        case value_type::double_: return std::to_string(std::any_cast<double>(a));
        case value_type::bool_: return std::any_cast<bool>(a) ? "true" : "false";
    }
    return {};
}

[[nodiscard]] inline auto any_equal(const std::any& a, const std::any& b, value_type vt) -> bool {
    switch (vt) {
        case value_type::string_: return std::any_cast<std::string>(a) == std::any_cast<std::string>(b);
        case value_type::int_: return std::any_cast<int>(a) == std::any_cast<int>(b);
        case value_type::long_long_: return std::any_cast<long long>(a) == std::any_cast<long long>(b);
        case value_type::double_: return std::any_cast<double>(a) == std::any_cast<double>(b);
        case value_type::bool_: return std::any_cast<bool>(a) == std::any_cast<bool>(b);
    }
    return false;
}

/// A token like `-3`, `-3.0`, `-.5` that should read as a value/positional, not
/// an option, when it doesn't name a registered flag.
[[nodiscard]] inline auto looks_like_number(std::string_view t) -> bool {
    if (t.size() < 2 || t[0] != '-') return false;
    return (std::isdigit(static_cast<unsigned char>(t[1])) != 0) || t[1] == '.';
}

}  // namespace detail

/// One option or positional. Returned by parser::add_argument for chaining.
class argument {
  public:
    explicit argument(std::vector<std::string> names) : names_(std::move(names)) {
        is_positional_ = names_.empty() || !names_.front().starts_with('-');
        canonical_ = names_.empty() ? std::string{} : names_.front();
        for (const auto& n : names_)
            if (n.starts_with("--")) { canonical_ = n; break; }
        metavar_ = default_metavar();
    }

    auto help(std::string h) -> argument& { help_ = std::move(h); return *this; }
    auto metavar(std::string m) -> argument& { metavar_ = std::move(m); return *this; }

    /// Set the value type and default in one go (type is inferred from `T`).
    /// `T` is normalized to the canonical stored type, so `default_value("x")`
    /// (a `const char*`) and `default_value(std::string("x"))` are equivalent.
    template <class T>
    auto default_value(T v) -> argument& {
        type_ = detail::type_tag<T>();
        default_ = to_canonical<T>(v);
        return *this;
    }

    /// Set the value type explicitly when there is no default to infer it from.
    template <class T>
    auto scan() -> argument& {
        type_ = detail::type_tag<T>();
        return *this;
    }

    /// A boolean store-true switch: present -> true, absent -> false.
    auto flag() -> argument& {
        is_flag_ = true;
        type_ = detail::value_type::bool_;
        default_ = std::any(false);
        return *this;
    }

    auto required() -> argument& { required_ = true; return *this; }
    auto hidden() -> argument& { hidden_ = true; return *this; }
    auto nargs(argparse::nargs n) -> argument& { nargs_ = n; return *this; }

    template <class T>
    auto choices(std::initializer_list<T> cs) -> argument& {
        type_ = detail::type_tag<T>();
        choices_.clear();
        for (const T& c : cs) choices_.emplace_back(to_canonical<T>(c));
        return *this;
    }

    [[nodiscard]] auto name() const -> const std::string& { return canonical_; }

  private:
    friend class parser;

    /// Normalize a user-supplied value of type `T` to the canonical std::any the
    /// parser stores/compares for that type (e.g. const char* -> std::string).
    template <class T>
    [[nodiscard]] static auto to_canonical(const T& c) -> std::any {
        constexpr detail::value_type vt = detail::type_tag<T>();
        if constexpr (vt == detail::value_type::string_) return std::any(std::string(c));
        else if constexpr (vt == detail::value_type::int_) return std::any(static_cast<int>(c));
        else if constexpr (vt == detail::value_type::long_long_) return std::any(static_cast<long long>(c));
        else if constexpr (vt == detail::value_type::double_) return std::any(static_cast<double>(c));
        else return std::any(static_cast<bool>(c));
    }

    [[nodiscard]] auto is_variadic() const -> bool {
        return nargs_ == nargs::zero_or_more || nargs_ == nargs::one_or_more;
    }

    [[nodiscard]] auto default_metavar() const -> std::string {
        std::string base = canonical_;
        while (base.starts_with('-')) base.erase(base.begin());
        for (char& c : base) c = (c == '-') ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return base;
    }

    /// The value to hand back from get(): the parsed one, else the default, else
    /// a type-appropriate zero so getters never see an empty std::any.
    [[nodiscard]] auto resolved() const -> std::any {
        if (value_.has_value()) return value_;
        if (default_.has_value()) return default_;
        if (is_variadic()) return std::any(std::vector<std::string>{});
        switch (type_) {
            case detail::value_type::string_: return std::any(std::string{});
            case detail::value_type::int_: return std::any(0);
            case detail::value_type::long_long_: return std::any(0LL);
            case detail::value_type::double_: return std::any(0.0);
            case detail::value_type::bool_: return std::any(false);
        }
        return std::any{};
    }

    [[nodiscard]] auto choices_str() const -> std::string {
        std::string out;
        for (std::size_t i = 0; i < choices_.size(); ++i)
            out += (i ? ", " : "") + detail::any_to_string(choices_[i], type_);
        return out;
    }

    auto enforce_choices(const std::any& v) const -> void {
        if (choices_.empty()) return;
        for (const auto& c : choices_)
            if (detail::any_equal(v, c, type_)) return;
        throw error("argument " + canonical_ + ": invalid choice '" +
                    detail::any_to_string(v, type_) + "' (choose from " + choices_str() + ")");
    }

    std::vector<std::string> names_;
    std::string canonical_, help_, metavar_;
    detail::value_type type_ = detail::value_type::string_;
    std::any default_, value_;
    std::vector<std::any> choices_;
    argparse::nargs nargs_ = nargs::one;
    bool is_positional_ = false;
    bool is_flag_ = false;
    bool required_ = false;
    bool hidden_ = false;
    bool is_help_ = false;
    bool used_ = false;
};

/// The parser: declare arguments, parse(), then get<T>() the values.
class parser {
  public:
    explicit parser(std::string prog, std::string description = {})
        : prog_(std::move(prog)), description_(std::move(description)) {
        add_argument("-h", "--help").flag().help("show this help message and exit").is_help_ = true;
    }

    /// Declare an argument. Names starting with `-` are options (any number of
    /// aliases); a bare name is a positional. Returns a reference for chaining.
    auto add_argument(std::initializer_list<std::string_view> names) -> argument& {
        std::vector<std::string> ns;
        for (auto n : names) ns.emplace_back(n);
        return add_impl(std::move(ns));
    }
    template <class... S>
    auto add_argument(S&&... names) -> argument& {
        return add_impl(std::vector<std::string>{std::string(std::forward<S>(names))...});
    }

    auto parse(int argc, const char* const* argv) -> void {
        parse(std::span<const char* const>(argv + (argc > 0 ? 1 : 0),
                                           static_cast<std::size_t>(argc > 0 ? argc - 1 : 0)));
    }

    auto parse(std::span<const char* const> tokens) -> void {
        std::vector<std::string> positional_tokens;
        bool no_more_opts = false;

        for (std::size_t i = 0; i < tokens.size(); ++i) {
            std::string tok = tokens[i];
            if (!no_more_opts && tok == "--") { no_more_opts = true; continue; }

            const bool is_opt = !no_more_opts && tok.size() > 1 && tok[0] == '-' &&
                                !detail::looks_like_number(tok);
            if (!is_opt) {
                positional_tokens.push_back(std::move(tok));
                continue;
            }

            std::string name = tok;
            std::optional<std::string> inline_val;
            if (tok.starts_with("--")) {
                if (const auto eq = tok.find('='); eq != std::string::npos) {
                    name = tok.substr(0, eq);
                    inline_val = tok.substr(eq + 1);
                }
            }

            argument* a = find(name);
            if (a == nullptr) throw error("unrecognized argument: " + name);
            if (a->is_help_) throw help_requested{};

            a->used_ = true;
            if (a->is_flag_) {
                if (inline_val) throw error("argument " + name + " takes no value");
                a->value_ = std::any(true);
            } else {
                std::string val;
                if (inline_val) {
                    val = *inline_val;
                } else {
                    if (i + 1 >= tokens.size()) throw error("argument " + name + ": expected a value");
                    val = tokens[++i];
                }
                std::any conv = detail::convert(val, a->type_, a->canonical_);
                a->enforce_choices(conv);
                a->value_ = std::move(conv);
            }
        }

        assign_positionals(positional_tokens);
        finalize();
    }

    template <class T>
    [[nodiscard]] auto get(std::string_view name) const -> T {
        return std::any_cast<T>(require(name)->resolved());
    }

    [[nodiscard]] auto is_used(std::string_view name) const -> bool {
        return require(name)->used_;
    }

    template <class T>
    [[nodiscard]] auto present(std::string_view name) const -> std::optional<T> {
        const argument* a = require(name);
        if (!a->used_) return std::nullopt;
        return std::any_cast<T>(a->resolved());
    }

    /// One-line `usage: ...` string, generated from the declared arguments.
    [[nodiscard]] auto usage() const -> std::string {
        std::string s = "usage: " + prog_;
        for (const auto& up : args_) {
            const argument* a = up.get();
            if (a->hidden_ || a->is_positional_) continue;
            const std::string tok = a->names_.front() + (a->is_flag_ ? "" : " " + a->metavar_);
            s += a->required_ ? " " + tok : " [" + tok + "]";
        }
        for (const auto* a : positionals_) {
            if (a->hidden_) continue;
            if (a->is_variadic())
                s += " " + a->metavar_ + " [" + a->metavar_ + " ...]";
            else if (a->nargs_ == nargs::optional)
                s += " [" + a->metavar_ + "]";
            else
                s += " " + a->metavar_;
        }
        return s + "\n";
    }

    /// Full `--help` text: usage, description, then the positional and option
    /// sections (hidden arguments excluded).
    [[nodiscard]] auto help_text() const -> std::string {
        std::string s = usage();
        if (!description_.empty()) s += "\n" + description_ + "\n";

        std::string pos, opt;
        for (const auto* a : positionals_) {
            if (a->hidden_) continue;
            pos += "  " + pad(a->metavar_) + a->help_ + "\n";
        }
        for (const auto& up : args_) {
            const argument* a = up.get();
            if (a->hidden_ || a->is_positional_) continue;
            std::string names;
            for (std::size_t i = 0; i < a->names_.size(); ++i)
                names += (i ? ", " : "") + a->names_[i];
            if (!a->is_flag_) names += " " + a->metavar_;
            opt += "  " + pad(names) + a->help_ + "\n";
        }
        if (!pos.empty()) s += "\npositional arguments:\n" + pos;
        if (!opt.empty()) s += "\noptions:\n" + opt;
        return s;
    }

  private:
    auto add_impl(std::vector<std::string> names) -> argument& {
        args_.push_back(std::make_unique<argument>(std::move(names)));
        argument* a = args_.back().get();
        if (a->is_positional_) {
            positionals_.push_back(a);
            lookup_.emplace(a->canonical_, a);
        } else {
            for (const auto& n : a->names_) lookup_.emplace(n, a);
        }
        return *a;
    }

    [[nodiscard]] auto find(const std::string& name) const -> argument* {
        const auto it = lookup_.find(name);
        return it == lookup_.end() ? nullptr : it->second;
    }

    /// Like find() but for the get()/is_used() API: an unknown name is a
    /// programmer error, not user input, so it throws std::out_of_range.
    [[nodiscard]] auto require(std::string_view name) const -> const argument* {
        const auto it = lookup_.find(std::string(name));
        if (it == lookup_.end())
            throw std::out_of_range("argparse: no such argument '" + std::string(name) + "'");
        return it->second;
    }

    auto assign_positionals(std::vector<std::string>& toks) -> void {
        std::size_t idx = 0;
        for (argument* a : positionals_) {
            if (a->is_variadic()) {
                std::vector<std::string> rest(toks.begin() + static_cast<std::ptrdiff_t>(idx),
                                              toks.end());
                idx = toks.size();
                if (a->nargs_ == nargs::one_or_more && rest.empty())
                    throw error("the following argument is required: " + a->metavar_);
                a->value_ = std::any(std::move(rest));
                a->used_ = !std::any_cast<const std::vector<std::string>&>(a->value_).empty();
            } else if (idx < toks.size()) {
                std::any conv = detail::convert(toks[idx++], a->type_, a->metavar_);
                a->enforce_choices(conv);
                a->value_ = std::move(conv);
                a->used_ = true;
            } else if (a->nargs_ == nargs::one) {
                throw error("the following argument is required: " + a->metavar_);
            }
        }
        if (idx < toks.size()) throw error("unrecognized argument: " + toks[idx]);
    }

    auto finalize() -> void {
        for (const auto& up : args_) {
            argument* a = up.get();
            if (a->value_.has_value() || a->is_help_) continue;
            if (a->required_ && !a->is_positional_)
                throw error("the following argument is required: " + a->canonical_);
            // leave value_ empty; resolved() supplies the default/zero on get()
        }
    }

    [[nodiscard]] static auto pad(const std::string& s) -> std::string {
        constexpr std::size_t col = 28;
        return s.size() + 2 >= col ? s + "  " : s + std::string(col - s.size(), ' ');
    }

    std::string prog_, description_;
    std::vector<std::unique_ptr<argument>> args_;  // owns; stable addresses
    std::vector<argument*> positionals_;           // in declaration order
    std::unordered_map<std::string, argument*> lookup_;  // every alias -> argument
};

}  // namespace argparse
