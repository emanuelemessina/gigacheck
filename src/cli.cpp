#include "cli.h"

CLI::CLI(std::string&& description)
    : name(""), description(description)
{
    opts.reserve(4);
}

void CLI::help()
{
    std::cout << "Usage:\n"
              << name << " [OPTIONS...]\n\n"
              << "Options:"
              << std::endl;
    for (auto pair : opts)
    {
        std::cout << "\t-" << pair.second.shortName << ", --" << pair.second.longName << "\t" << pair.second.description << std::endl;
    }
}

CLI& CLI::option(Option&& opt)
{
    opts.emplace(opt.longName, opt);
    return *this;
}

void CLI::parse(int argc, char* argv[])
{
    name = argv[0];

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        options::iterator it;
        if (arg.rfind("--", 0) == 0)
        { // Long option
            std::string optionName = arg.substr(2);
            it = opts.find(optionName);
        }
        else if (arg.rfind("-", 0) == 0)
        { // Short option
            std::string optionName = arg.substr(1);
            it = std::find_if(opts.begin(), opts.end(), [&](std::pair<const std::string, Option>& pair)
                              { return pair.second.shortName == optionName; });
        }
        else
        {
            it = opts.end();
        }

        if (it != opts.end())
        {
            // Check if the next argument (if there is one) is the value
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                std::string valueStr = argv[++i];
                it->second.setValue(valueStr);
                continue;
            }

            it->second.setDefaultValue();
        }
    }
}

Option& CLI::get(std::string&& longName)
{
    auto it = opts.find(longName);
    return it->second;
}

Option* Option::setValue(const std::string& valueStr)
{
    if (std::holds_alternative<int>(value))
    {
        value = std::stoi(valueStr);
    }
    else if (std::holds_alternative<std::string>(value))
    {
        value = valueStr;
    }
    else if (std::holds_alternative<bool>(value))
    {
        value = (valueStr == "true" || valueStr == "1");
    }

    set = true;

    return this;
}

Option* Option::setDefaultValue()
{
    set = true;

    return this;
}