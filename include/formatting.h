#include <iomanip>
#include <iostream>
#include <sstream>
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
