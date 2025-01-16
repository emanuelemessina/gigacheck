#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#define OPTION_INT_UNSET -1
#define OPTION_STRING_UNSET std::string("")
#define OPTION_BOOL_UNSET false

/**
 * @brief Concept to validate option types for CLI.
 * Valid option types are int, std::string, and bool.
 */
template <typename T>
concept ValidOptionType = std::is_same_v<T, int> ||
                          std::is_same_v<T, std::string> ||
                          std::is_same_v<T, bool>;

typedef std::variant<int, std::string, bool> option_value;

class CLI;

class Option
{
    std::string shortName; ///< Short name for the option (e.g., '-o').
    std::string longName;  ///< Long name for the option (e.g., '--option').
    option_value value;
    std::string description;
    bool set;

  public:
    /**
     * @brief Constructs an Option with specified short name, long name, and default value.
     * @tparam T Type of the default value, must be a valid option type.
     * @param shortName Short name for the option.
     * @param longName Long name for the option.
     * @param defaultValue Default value for the option.
     */
    template <ValidOptionType T>
    Option(const std::string&& shortName, const std::string&& longName, T&& defaultValue, const std::string&& description = "")
        : shortName(shortName), longName(longName), value(defaultValue), description(description), set(false) {}
    /**
     * @brief Sets the value of the option based on the provided string.
     * @param valueStr String representation of the value to set.
     * @return Pointer to the current Option instance.
     */
    Option* setValue(const std::string& valueStr);

    Option* setDefaultValue();

    /**
     * @brief Returns the casted value of the option. Panics if the type to cast to is wrong.
     *
     * @tparam T
     * @return T
     */
    template <typename T>
    inline T getValue()
    {
        return *(std::get_if<T>(&value));
    }

    inline bool isSet()
    {
        return set;
    }

    friend class CLI;
};

typedef std::unordered_map<std::string, Option> options;

class CLI
{
  public:
    /**
     * @brief Constructs a new CLI with the specified description.
     *
     * @param description Description of the CLI.
     */
    CLI(std::string&& description = "");

    CLI& option(Option&& opt);

    /**
     * @brief Parses the command line arguments and sets the values for recognized options. Fails if an unrecognized argument is found, in which case in prints it, followed by the help.
     * @param argc
     * @param argv
     * @return false if there was an unrecognized argument.
     */
    bool parse(int argc, char* argv[]);

    Option& get(std::string&& name);

    void help();

  private:
    std::string name;
    std::string description;
    options opts;
};

inline void clamp_int_argument(int& arg, int min, int max)
{
    if (arg < min)
    {
        std::cout << std::format("Using min size {}\n", min) << std::endl;
        arg = min;
    }
    else if (arg > max)
    {
        std::cout << std::format("Capping to max size {}\n", max) << std::endl;
        arg = max;
    }
}
