// Logger implementation — thread-safe stdout logger with INFO/DEBUG/WARNING/ERROR levels.
// Level is controlled at runtime via the LOG_LEVEL environment variable.

#include "logger.hpp"

namespace app {
namespace utils {

LogLevel Logger::current_level_ = LogLevel::INFO;

void Logger::set_level(LogLevel level)
{
    current_level_ = level;
}

void Logger::set_level_from_env()
{
    const char* val = std::getenv("LOG_LEVEL");
    if (!val)
        return;
    std::string s(val);
    if (s == "DEBUG")
        current_level_ = LogLevel::DEBUG;
    else if (s == "INFO")
        current_level_ = LogLevel::INFO;
    else if (s == "WARNING")
        current_level_ = LogLevel::WARNING;
    else if (s == "ERROR")
        current_level_ = LogLevel::ERROR;
}

std::string Logger::get_timestamp()
{
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level)
{
    switch (level)
    {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& message)
{
    if (level < current_level_)
        return;

    std::ostream& stream = (level >= LogLevel::ERROR) ? std::cerr : std::cout;
    stream << "[" << get_timestamp() << "] "
           << "[" << level_to_string(level) << "] " << message << std::endl;
}

void Logger::debug(const std::string& message)
{
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message)
{
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message)
{
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message)
{
    log(LogLevel::ERROR, message);
}

}  // namespace utils
}  // namespace app
