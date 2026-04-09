// AWS S3 — aws.py parity (upload direct URL, download_from_s3) +
// PermanentRedirect retry and per-call SDK client pattern.
#include "aws.hpp"

#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>

#include <filesystem>
#include <fstream>

#include "../config.hpp"
#include "logger.hpp"

namespace app {
namespace utils {

namespace {

template <typename Err>
std::string redirect_region(const Err& err)
{
    if (std::string(err.GetExceptionName().c_str()) != "PermanentRedirect")
        return {};
    const auto& headers = err.GetResponseHeaders();
    auto it = headers.find("x-amz-bucket-region");
    if (it != headers.end() && !it->second.empty())
        return std::string(it->second.c_str());
    return {};
}

}  // namespace

S3Client::S3Client()
{
    auto& config = app::config::AppConfig::getInstance();
    bucket_ = config.aws_bucket;
    region_ = config.aws_region;
}

S3Client::~S3Client() = default;

std::optional<std::string> S3Client::upload_bytes_and_get_url(const void* data, size_t size,
                                                              const std::string& object_name,
                                                              const std::string& content_type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string effectiveRegion = region_;

    for (int attempt = 0; attempt <= 1; ++attempt)
    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.region = effectiveRegion;
        clientConfig.maxConnections = 20;
        clientConfig.followRedirects = Aws::Client::FollowRedirectsPolicy::ALWAYS;
        Aws::S3::S3Client s3_client(clientConfig);

        Aws::S3::Model::PutObjectRequest request;
        request.SetBucket(bucket_);
        request.SetKey(object_name);
        request.SetContentType(content_type);

        auto stream = Aws::MakeShared<Aws::StringStream>("S3Upload");
        stream->write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        request.SetBody(stream);

        auto outcome = s3_client.PutObject(request);

        if (outcome.IsSuccess())
        {
            std::string url =
                "https://" + bucket_ + ".s3." + effectiveRegion + ".amazonaws.com/" + object_name;
            app::utils::Logger::info("[S3Client] Image uploaded successfully to S3 bucket=" +
                                     bucket_ + " object_name=" + object_name +
                                     " size_bytes=" + std::to_string(size) + " url=" + url);
            region_ = effectiveRegion;
            return url;
        }

        std::string redir = redirect_region(outcome.GetError());
        if (!redir.empty() && attempt == 0)
        {
            app::utils::Logger::info("[S3Client] PermanentRedirect: retrying with region " + redir +
                                     " (was " + effectiveRegion + ")");
            effectiveRegion = redir;
            continue;
        }

        app::utils::Logger::error("[S3Client] Error uploading image to S3 bucket=" + bucket_ +
                                  " object_name=" + object_name +
                                  " error=" + std::string(outcome.GetError().GetExceptionName()) +
                                  " - " + outcome.GetError().GetMessage());
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<std::string> S3Client::upload_video_file_and_get_url(const std::string& file_path,
                                                                   const std::string& object_name,
                                                                   const std::string& content_type)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::ifstream file_check(file_path);
    if (!file_check.good())
    {
        app::utils::Logger::error("[S3Client] Video file not found: " + file_path);
        return std::nullopt;
    }
    file_check.close();

    std::string effectiveRegion = region_;

    for (int attempt = 0; attempt <= 1; ++attempt)
    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.region = effectiveRegion;
        clientConfig.maxConnections = 20;
        clientConfig.followRedirects = Aws::Client::FollowRedirectsPolicy::ALWAYS;
        Aws::S3::S3Client s3_client(clientConfig);

        Aws::S3::Model::PutObjectRequest request;
        request.SetBucket(bucket_);
        request.SetKey(object_name);
        request.SetContentType(content_type);

        std::shared_ptr<Aws::IOStream> input_data = Aws::MakeShared<Aws::FStream>(
            "S3UploadAllocation", file_path.c_str(), std::ios_base::in | std::ios_base::binary);

        if (!input_data->good())
        {
            app::utils::Logger::error("[S3Client] Failed to open file stream for: " + file_path);
            return std::nullopt;
        }

        request.SetBody(input_data);

        auto outcome = s3_client.PutObject(request);

        if (outcome.IsSuccess())
        {
            std::string url =
                "https://" + bucket_ + ".s3." + effectiveRegion + ".amazonaws.com/" + object_name;
            app::utils::Logger::info("[S3Client] File uploaded successfully to S3 bucket=" +
                                     bucket_ + " object_name=" + object_name + " url=" + url);
            region_ = effectiveRegion;
            return url;
        }

        std::string redir = redirect_region(outcome.GetError());
        if (!redir.empty() && attempt == 0)
        {
            app::utils::Logger::info("[S3Client] PermanentRedirect: retrying with region " + redir +
                                     " (was " + effectiveRegion + ")");
            effectiveRegion = redir;
            continue;
        }

        app::utils::Logger::error(
            "[S3Client] S3 upload failed: " + std::string(outcome.GetError().GetExceptionName()) +
            " - " + outcome.GetError().GetMessage());
        return std::nullopt;
    }
    return std::nullopt;
}

std::string S3Client::download_from_s3(const std::string& local_path, const std::string& s3_key,
                                       const std::optional<std::string>& fallback)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (s3_key.empty())
    {
        if (fallback.has_value())
            return *fallback;
        throw std::invalid_argument("S3 key required");
    }

    namespace fs = std::filesystem;
    if (fs::exists(local_path))
    {
        app::utils::Logger::info("[S3Client] File exists locally model_path=" + local_path);
        return local_path;
    }

    fs::path lp(local_path);
    if (lp.has_parent_path())
    {
        std::error_code ec;
        fs::create_directories(lp.parent_path(), ec);
    }

    app::utils::Logger::info("[S3Client] Downloading from S3 bucket=" + bucket_ + " key=" + s3_key +
                             " local_path=" + local_path);

    std::string effectiveRegion = region_;

    for (int attempt = 0; attempt <= 1; ++attempt)
    {
        Aws::Client::ClientConfiguration clientConfig;
        clientConfig.region = effectiveRegion;
        clientConfig.maxConnections = 20;
        clientConfig.followRedirects = Aws::Client::FollowRedirectsPolicy::ALWAYS;
        Aws::S3::S3Client s3_client(clientConfig);

        Aws::S3::Model::GetObjectRequest request;
        request.SetBucket(bucket_);
        request.SetKey(s3_key);

        auto outcome = s3_client.GetObject(request);

        if (outcome.IsSuccess())
        {
            std::ofstream outfile(local_path, std::ios::binary);
            outfile << outcome.GetResult().GetBody().rdbuf();
            outfile.close();
            if (!outfile)
            {
                app::utils::Logger::error("[S3Client] Download failed local_path=" + local_path +
                                          " error=write_failed");
                throw std::runtime_error("Download failed: could not write " + local_path);
            }
            app::utils::Logger::info("[S3Client] Downloaded successfully local_path=" + local_path);
            region_ = effectiveRegion;
            return local_path;
        }

        std::string redir = redirect_region(outcome.GetError());
        if (!redir.empty() && attempt == 0)
        {
            app::utils::Logger::info(
                "[S3Client] PermanentRedirect (GetObject): retrying with region " + redir +
                " (was " + effectiveRegion + ")");
            effectiveRegion = redir;
            continue;
        }

        std::string err = std::string(outcome.GetError().GetExceptionName()) + " - " +
                          outcome.GetError().GetMessage();
        app::utils::Logger::error("[S3Client] Download failed key=" + s3_key + " error=" + err);
        throw std::runtime_error("Download failed: " + err);
    }

    throw std::runtime_error("Download failed: PermanentRedirect retry exhausted");
}

}  // namespace utils
}  // namespace app
