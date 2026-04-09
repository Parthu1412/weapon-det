// Thread-safe Logger singleton — INFO / DEBUG / WARNING / ERROR levels.
// Runtime level selection via the LOG_LEVEL environment variable.
// Implementation in logger.cpp.
#pragma once

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace app {
namespace utils {

enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };

class Logger
{
public:
    static void set_level(LogLevel level);
    static void set_level_from_env();  // Reads LOG_LEVEL from environment

    static void log(LogLevel level, const std::string& message);
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);

private:
    static LogLevel current_level_;
    static std::string get_timestamp();
    static std::string level_to_string(LogLevel level);
};

}  // namespace utils
}  // namespace app
