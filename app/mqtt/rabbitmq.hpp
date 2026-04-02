// RabbitMQ client — publishes weapon notification messages to a queue via AMQP.
// Uses SimpleAmqpClient and reconnects on stale connections, matching fall-cpp
// rabbitmq.hpp structure and weapon-detection app/mqtt/rabitmq.py behaviour.

#pragma once

#include <SimpleAmqpClient/SimpleAmqpClient.h>

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <sys/time.h>
#endif

#include "../config.hpp"
#include "../utils/logger.hpp"
#include "../utils/message.hpp"

using json = nlohmann::json;

namespace app {
namespace mqtt {

namespace {

// Python pika uses socket_timeout on the connection; SimpleAmqpClient has no direct equivalent,
// so we set SO_RCVTIMEO/SO_SNDTIMEO on the broker socket after connect.
void apply_amqp_socket_timeout(AmqpClient::Channel::ptr_t& channel, int timeout_sec)
{
    if (!channel || timeout_sec <= 0)
        return;
    int fd = channel->GetSocketFD();
    if (fd < 0)
        return;
#ifdef _WIN32
    DWORD ms = static_cast<DWORD>(timeout_sec) * 1000u;
    setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&ms), static_cast<int>(sizeof(ms)));
    setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&ms), static_cast<int>(sizeof(ms)));
#else
    struct timeval tv {};
    tv.tv_sec = timeout_sec;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

}  // namespace

class RabbitMQClient
{
public:
    RabbitMQClient()
    {
        auto& config = app::config::AppConfig::getInstance();
        max_retries_ = config.max_retries;
    }

    // Single connection attempt; throws on failure (connect_with_retry handles retries).
    void connect()
    {
        auto& config = app::config::AppConfig::getInstance();
        if (config.rabbitmq_host.empty())
        {
            app::utils::Logger::info(
                "[RabbitMQ] RABBITMQ_HOST_PRODUCER / RABBITMQ_HOST not set – RabbitMQ publishing "
                "disabled.");
            return;
        }
        if (config.rabbitmq_use_ssl)
        {
            try
            {
                const char* ca_path = "/etc/ssl/certs/ca-certificates.crt";
                channel_ = AmqpClient::Channel::CreateSecure(
                    ca_path, config.rabbitmq_host, "", "", config.rabbitmq_port,
                    config.rabbitmq_user, config.rabbitmq_pass, "/", 131072,
                    false  // verify_hostname_and_peer = false (matches Python CERT_NONE)
                );
            } catch (const std::exception& e)
            {
                std::string err(e.what());
                if (err.find("SSL support") != std::string::npos ||
                    err.find("SSL") != std::string::npos)
                {
                    app::utils::Logger::error(
                        "[RabbitMQ] RABBITMQ_USE_SSL=1 but SimpleAmqpClient was built "
                        "without SSL. Trying non-SSL fallback.");
                    channel_ =
                        AmqpClient::Channel::Create(config.rabbitmq_host, config.rabbitmq_port,
                                                    config.rabbitmq_user, config.rabbitmq_pass);
                } else
                {
                    throw;
                }
            }
        } else
        {
            channel_ = AmqpClient::Channel::Create(config.rabbitmq_host, config.rabbitmq_port,
                                                   config.rabbitmq_user, config.rabbitmq_pass);
        }
        if (channel_)
            apply_amqp_socket_timeout(channel_, config.rabbitmq_socket_timeout_sec);
        app::utils::Logger::info("[RabbitMQ] Connected to " + config.rabbitmq_host + ":" +
                                 std::to_string(config.rabbitmq_port) +
                                 (config.rabbitmq_use_ssl ? " (SSL)" : ""));
    }

    void connect_with_retry()
    {
        auto& config = app::config::AppConfig::getInstance();
        if (config.rabbitmq_host.empty())
            return;

        int retry_count = 0;
        while (retry_count < max_retries_)
        {
            try
            {
                connect();
                app::utils::Logger::info("[RabbitMQ] Connection successful");
                return;
            } catch (const std::exception& e)
            {
                retry_count++;
                app::utils::Logger::error("[RabbitMQ] Connection failed (retry " +
                                          std::to_string(retry_count) + "/" +
                                          std::to_string(max_retries_) + "): " + e.what());
                if (retry_count < max_retries_)
                    std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        app::utils::Logger::error("[RabbitMQ] Max retries reached. Exiting application.");
        std::exit(1);
    }

    bool is_connected() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return channel_ != nullptr;
    }

    void publish(const std::string& queue_name, const app::utils::WeaponMessage& message)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!channel_)
        {
            app::utils::Logger::warning(
                "[RabbitMQ] Cannot publish, channel is null. Reconnecting...");
            connect_with_retry();
            if (!channel_)
                return;
        }

        json j;
        app::utils::to_json(j, message);
        std::string payload = j.dump();

        try
        {
            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg = AmqpClient::BasicMessage::Create(payload);
            amqp_msg->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg);
            app::utils::Logger::info("[RabbitMQ] Message sent to queue | queue_name=" + queue_name +
                                     " | message_size=" + std::to_string(payload.size()));

        } catch (const std::exception& e)
        {
            app::utils::Logger::warning(
                "[RabbitMQ] Publish failed due to stale connection, reconnecting and retrying | "
                "error=" +
                std::string(e.what()) + " | queue_name=" + queue_name);
            channel_ = nullptr;
            connect_with_retry();

            channel_->DeclareQueue(queue_name, false, true, false, false);
            auto amqp_msg2 = AmqpClient::BasicMessage::Create(payload);
            amqp_msg2->DeliveryMode(AmqpClient::BasicMessage::dm_persistent);
            channel_->BasicPublish("", queue_name, amqp_msg2);
            app::utils::Logger::info(
                "[RabbitMQ] Message sent to queue after reconnect | queue_name=" + queue_name +
                " | message_size=" + std::to_string(payload.size()));
        }
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        channel_ = nullptr;
        app::utils::Logger::info("[RabbitMQ] Connection closed");
    }

private:
    mutable std::mutex mutex_;
    AmqpClient::Channel::ptr_t channel_;
    int max_retries_ = 3;
};

}  // namespace mqtt
}  // namespace app
