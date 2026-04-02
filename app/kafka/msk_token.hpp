// Interface for the MSK IAM token provider.
// Declares MskToken::generate(), which fetches an AWS SigV4-signed
// OAUTHBEARER token string for use with librdkafka SASL authentication.

#pragma once

#include <string>

namespace app {
namespace kafka {

/**
 * Generate an AWS MSK IAM OAuth token using SigV4 presigning.
 * Matches the Python aws-msk-iam-sasl-signer behavior:
 * GET https://kafka.{region}.amazonaws.com/?Action=kafka-cluster:Connect
 * Presign with SigV4, base64 URL-safe encode the signed URL (no padding).
 *
 * @param region AWS region (e.g. "us-east-1")
 * @return Token string on success, empty string on failure
 */
std::string generate_msk_iam_token(const std::string& region);

}  // namespace kafka
}  // namespace app
