// Kafka producer — matches fall-cpp layout: MSK/OAUTHBEARER, background poll thread, mutex on produce.

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "../utils/message.hpp"

namespace app::kafka {

class KafkaProducer
{
public:
    KafkaProducer();
    ~KafkaProducer();

    /** Connects producer; retries AppConfig::max_retries on failure. Exits process if all fail. */
    void start_with_retry();
    void stop();

    void produce(const std::string& topic, const app::utils::WeaponMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mutex_;
};

}  // namespace app::kafka
