#pragma once

#include <memory>
#include <string>
#include <optional>
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>

namespace app {
namespace utils {

class S3Client {
public:
    S3Client();
    ~S3Client();

    // Prevent copying because of the underlying AWS Client
    S3Client(const S3Client&) = delete;
    S3Client& operator=(const S3Client&) = delete;

    /**
     * Uploads raw bytes (e.g. image) to S3 and returns the public URL.
     * @param data Raw bytes to upload
     * @param object_name Destination key in the S3 bucket
     * @param content_type MIME type (default image/jpeg)
     * @return The S3 URL if successful, std::nullopt otherwise
     */
    std::optional<std::string> upload_bytes_and_get_url(
        const void* data,
        size_t size,
        const std::string& object_name,
        const std::string& content_type = "image/jpeg"
    );

    /**
     * Uploads a video file to S3 and returns the public URL.
     * @param file_path Local path to the .mp4 file
     * @param object_name Destination key in the S3 bucket
     * @param content_type MIME type of the file
     * @return The S3 URL if successful, std::nullopt otherwise
     */
    std::optional<std::string> upload_video_file_and_get_url(
        const std::string& file_path, 
        const std::string& object_name, 
        const std::string& content_type = "video/mp4"
    );

private:
    std::string bucket_;
    std::string region_;
    // Match Python: boto3 client is created per call (no shared client member)
};

// Helper struct to manage AWS SDK Lifecycle globally
struct AwsApiManager {
    AwsApiManager() {
        Aws::InitAPI(options);
    }
    ~AwsApiManager() {
        Aws::ShutdownAPI(options);
    }
    Aws::SDKOptions options;
};

} // namespace utils
} // namespace app