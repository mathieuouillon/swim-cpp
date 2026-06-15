//******************************************************************************
//* HIPO reading core — error types                                            *
//* Part of the modern C++23 reading API (see hipo4/hipo.hpp).                 *
//******************************************************************************
#pragma once

#include <cstdint>
#include <expected>
#include <stdexcept>
#include <string>
#include <utility>

namespace hipo {

/// Error categories produced by the reading core.
enum class errc : std::uint8_t {
    cannot_open,
    not_a_hipo_file,
    truncated_file,
    bad_record_header,
    decompress_failed,
    unsupported_compression,
    no_dictionary,
    unknown_bank,
    unknown_column,
    type_mismatch,
    event_out_of_range,
    write_failed,
    bad_usage,
};

[[nodiscard]] constexpr std::string_view to_string(errc c) noexcept {
    switch (c) {
        case errc::cannot_open:             return "cannot_open";
        case errc::not_a_hipo_file:         return "not_a_hipo_file";
        case errc::truncated_file:          return "truncated_file";
        case errc::bad_record_header:       return "bad_record_header";
        case errc::decompress_failed:       return "decompress_failed";
        case errc::unsupported_compression: return "unsupported_compression";
        case errc::no_dictionary:           return "no_dictionary";
        case errc::unknown_bank:            return "unknown_bank";
        case errc::unknown_column:          return "unknown_column";
        case errc::type_mismatch:           return "type_mismatch";
        case errc::event_out_of_range:      return "event_out_of_range";
        case errc::write_failed:            return "write_failed";
        case errc::bad_usage:               return "bad_usage";
    }
    return "unknown";
}

/// An error value: category plus a pre-formatted human-readable message.
struct error {
    errc        code{};
    std::string message;
};

/// The expected/error result type used by fallible operations of the
/// reading core (open, name resolution, random access, record decode).
template <class T>
using result = std::expected<T, error>;

/// Exception thrown when corruption is detected in the middle of an
/// iteration (where returning an expected from every `operator++` would
/// pollute the hot loop), and by the throwing `hipo::file` constructor.
class io_error final : public std::runtime_error {
  public:
    explicit io_error(error e) : std::runtime_error(e.message), err_(std::move(e)) {}
    [[nodiscard]] const error& detail() const noexcept { return err_; }

  private:
    error err_;
};

} // namespace hipo
