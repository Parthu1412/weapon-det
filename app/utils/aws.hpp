// Interface for the AWS S3 client — matches weapon-detection app/utils/aws.py (S3Client)
// plus fall-cpp upload robustness (redirect retry). Kafka MSK token provider lives in kafka/.

#pragma once

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace app {
namespace utils {

class S3Client
{
public:
    S3Client();
    ~S3Client();

    S3Client(const S3Client&) = delete;
    S3Client& operator=(const S3Client&) = delete;

    /** Same as Python upload_bytes_and_get_direct_url (returns URL or nullopt on failure). */
    std::optional<std::string> upload_bytes_and_get_url(const void* data, size_t size,
                                                          const std::string& object_name,
                                                          const std::string& content_type = "image/jpeg");

    /** Alias for Python naming. */
    std::optional<std::string> upload_bytes_and_get_direct_url(const void* data, size_t size,
                                                               const std::string& object_name,
                                                               const std::string& content_type = "image/jpeg")
    {
        return upload_bytes_and_get_url(data, size, object_name, content_type);
    }

    std::optional<std::string> upload_video_file_and_get_url(
        const std::string& file_path, const std::string& object_name,
        const std::string& content_type = "video/mp4");

    /**
     * Download s3_key to local_path if missing. If s3_key is empty and fallback is set, returns
     * fallback. Otherwise throws std::invalid_argument. On download failure throws std::runtime_error
     * (same as Python RuntimeError).
     */
    std::string download_from_s3(const std::string& local_path, const std::string& s3_key,
                                 const std::optional<std::string>& fallback = std::nullopt);

private:
    mutable std::mutex mutex_;
    std::string bucket_;
    std::string region_;
};

struct AwsApiManager {
    AwsApiManager()
    {
        Aws::InitAPI(options);
    }
    ~AwsApiManager()
    {
        Aws::ShutdownAPI(options);
    }
    Aws::SDKOptions options;
};

}  // namespace utils
}  // namespace app
