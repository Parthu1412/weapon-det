#include "aws.hpp"
#include "../config.hpp"
#include "logger.hpp"
#include <fstream>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

namespace app {
namespace utils {

S3Client::S3Client() {
    auto& config = app::config::AppConfig::getInstance();
    bucket_ = config.aws_bucket;
    region_ = config.aws_region;
    // Match Python: no shared client — boto3 client is created fresh per upload call
}

S3Client::~S3Client() = default;

std::optional<std::string> S3Client::upload_bytes_and_get_url(
    const void* data,
    size_t size,
    const std::string& object_name,
    const std::string& content_type)
{
    // Match Python: boto3.client("s3", ...) created per call
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.region = region_;
    clientConfig.maxConnections = 20;
    Aws::S3::S3Client s3_client(clientConfig);

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket_);
    request.SetKey(object_name);
    request.SetContentType(content_type);

    auto stream = Aws::MakeShared<Aws::StringStream>("S3Upload");
    stream->write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    request.SetBody(stream);

    auto outcome = s3_client.PutObject(request);

    if (outcome.IsSuccess()) {
        std::string url = "https://" + bucket_ + ".s3." + region_ + ".amazonaws.com/" + object_name;
        app::utils::Logger::info("[S3Client] Uploaded bytes to S3: " + url);
        return url;
    } else {
        app::utils::Logger::error("[S3Client] S3 upload failed: " +
            std::string(outcome.GetError().GetExceptionName()) + " - " +
            outcome.GetError().GetMessage());
        return std::nullopt;
    }
}

std::optional<std::string> S3Client::upload_video_file_and_get_url(
    const std::string& file_path, 
    const std::string& object_name, 
    const std::string& content_type) 
{
    // Verify file exists and is readable
    std::ifstream file_check(file_path);
    if (!file_check.good()) {
        app::utils::Logger::error("[S3Client] Video file not found: " + file_path);
        return std::nullopt;
    }
    file_check.close();

    // Match Python: boto3.client("s3", ...) created per call
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.region = region_;
    clientConfig.maxConnections = 20;
    Aws::S3::S3Client s3_client(clientConfig);

    Aws::S3::Model::PutObjectRequest request;
    request.SetBucket(bucket_);
    request.SetKey(object_name);
    request.SetContentType(content_type);

    // Stream the file directly from disk to AWS
    std::shared_ptr<Aws::IOStream> input_data = 
        Aws::MakeShared<Aws::FStream>("S3UploadAllocation", 
                                      file_path.c_str(), 
                                      std::ios_base::in | std::ios_base::binary);

    if (!input_data->good()) {
        app::utils::Logger::error("[S3Client] Failed to open file stream for: " + file_path);
        return std::nullopt;
    }

    request.SetBody(input_data);

    // Execute the upload synchronously (Thread is already detached in msg_gen.cpp)
    auto outcome = s3_client.PutObject(request);

    if (outcome.IsSuccess()) {
        std::string url = "https://" + bucket_ + ".s3." + region_ + ".amazonaws.com/" + object_name;
        app::utils::Logger::info("[S3Client] Uploaded video file to S3: " + url);
        return url;
    } else {
        app::utils::Logger::error("[S3Client] S3 upload failed: " +
            std::string(outcome.GetError().GetExceptionName()) + " - " +
            outcome.GetError().GetMessage());
        return std::nullopt;
    }
}

} // namespace utils
} // namespace app