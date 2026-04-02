#pragma once

#include "../utils/message.hpp"
#include <memory>
#include <string>

namespace app::kafka {

class KafkaProducer {
public:
    KafkaProducer();
    ~KafkaProducer();

    void start_with_retry();
    void stop();

    void produce(const std::string& topic, const app::utils::WeaponMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace app::kafka
