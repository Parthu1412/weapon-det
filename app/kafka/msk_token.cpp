// MSK IAM token provider — generates short-lived SASL/OAUTHBEARER tokens for
// authenticating to AWS MSK clusters using IAM credentials (instance role,
// environment variables, or credential chain), replacing static username/password.

#include "msk_token.hpp"

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/auth/signer/AWSAuthV4Signer.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/http/HttpTypes.h>
#include <aws/core/http/URI.h>
#include <aws/core/utils/Array.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/memory/AWSMemory.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

#include <algorithm>
#include <memory>

namespace app {
namespace kafka {

namespace {

// Convert standard base64 to URL-safe base64 and strip padding (matches Python
// urlsafe_b64encode().rstrip('='))
std::string to_urlsafe_base64_no_padding(const std::string& b64)
{
    std::string result = b64;
    std::replace(result.begin(), result.end(), '+', '-');
    std::replace(result.begin(), result.end(), '/', '_');
    // Strip trailing '='
    while (!result.empty() && result.back() == '=')
    {
        result.pop_back();
    }
    return result;
}

}  // namespace

std::string generate_msk_iam_token(const std::string& region)
{
    Aws::String regionStr(region.c_str());
    Aws::String host = "kafka." + regionStr + ".amazonaws.com";
    Aws::Http::URI uri;
    uri.SetScheme(Aws::Http::Scheme::HTTPS);
    uri.SetAuthority(host);
    uri.SetPath("/");
    uri.AddQueryStringParameter("Action", "kafka-cluster:Connect");

    Aws::IOStreamFactory streamFactory = []() { return Aws::New<Aws::StringStream>("msk_token"); };

    std::shared_ptr<Aws::Http::HttpRequest> request =
        Aws::Http::CreateHttpRequest(uri, Aws::Http::HttpMethod::HTTP_GET, streamFactory);

    auto credentialsProvider = std::make_shared<Aws::Auth::DefaultAWSCredentialsProviderChain>();
    Aws::Client::AWSAuthV4Signer signer(credentialsProvider, "kafka-cluster", regionStr,
                                        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never);

    constexpr long long expirationSeconds = 900;  // 15 minutes (matches AWS MSK)
    if (!signer.PresignRequest(*request, region.c_str(), "kafka-cluster", expirationSeconds))
    {
        return "";
    }

    Aws::String signedUri = request->GetURIString(true);
    if (signedUri.empty())
    {
        return "";
    }

    // Append User-Agent query param — matches Python aws-msk-iam-sasl-signer behavior
    std::string finalUri = std::string(signedUri.c_str(), signedUri.size());
    finalUri += "&User-Agent=aws-msk-iam-sasl-signer-cpp%2F1.0.0";

    Aws::Utils::ByteBuffer buf(reinterpret_cast<const unsigned char*>(finalUri.c_str()),
                               finalUri.size());
    Aws::String b64 = Aws::Utils::HashingUtils::Base64Encode(buf);
    return to_urlsafe_base64_no_padding(std::string(b64.c_str(), b64.size()));
}

}  // namespace kafka
}  // namespace app
