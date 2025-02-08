#pragma once

#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

inline std::string humanReadableMemSize(size_t bytes)
{
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    const size_t numUnits = sizeof(units) / sizeof(units[0]);

    double size = static_cast<double>(bytes);
    size_t unitIndex = 0;

    // Scale size to the largest possible unit
    while (size >= 1024 && unitIndex < numUnits - 1)
    {
        size /= 1024.0;
        ++unitIndex;
    }

    // Format the result with two decimal points
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
    return oss.str();
}

inline size_t parseMemSizeString(const std::string& input)
{
    const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    const size_t numUnits = sizeof(units) / sizeof(units[0]);
    double size = 0.0;
    std::string unit;

    // Read the numeric value and unit from the input string
    std::istringstream iss(input);
    iss >> size >> unit;

    if (iss.fail() || size < 0)
    {
        throw std::invalid_argument("Invalid input format. Expected format: '<number> <unit>'");
    }

    // Normalize the unit to uppercase for comparison
    for (auto& c : unit)
    {
        c = std::toupper(c);
    }

    // Find the unit in the predefined array
    size_t unitIndex = numUnits; // Initialize to an invalid index
    for (size_t i = 0; i < numUnits; ++i)
    {
        if (unit == units[i])
        {
            unitIndex = i;
            break;
        }
    }

    if (unitIndex == numUnits)
    {
        throw std::invalid_argument("Invalid unit. Valid units are B, KB, MB, GB, TB, PB.");
    }

    // Convert the size to bytes
    for (size_t i = 0; i < unitIndex; ++i)
    {
        size *= 1024.0;
    }

    // Return the size as a size_t
    return static_cast<size_t>(size);
}
