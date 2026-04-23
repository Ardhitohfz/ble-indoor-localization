#pragma once
#include <NimBLEDevice.h>
#include <atomic>
#include "telemetry.h"

class GenericServerCallbacks : public NimBLEServerCallbacks
{
protected:
    const char *deviceRole;
    std::atomic<bool> *connectionFlag;
    uint32_t *connectionCounter;

    virtual void applyConnectionParams(NimBLEServer *pServer, NimBLEConnInfo &connInfo)
    {
        pServer->updateConnParams(
            connInfo.getConnHandle(),
            16,
            32,
            0,
            400);
    }

    virtual void onConnectHook(NimBLEServer *pServer, NimBLEConnInfo &connInfo) {}
    virtual void onDisconnectHook(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) {}

public:
    GenericServerCallbacks(const char *role, std::atomic<bool> *connFlag, uint32_t *connCounter = nullptr)
        : deviceRole(role), connectionFlag(connFlag), connectionCounter(connCounter) {}

    void onConnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override
    {
        *connectionFlag = true;

        if (connectionCounter != nullptr)
        {
            (*connectionCounter)++;
        }

        LOG_I(deviceRole, "Client connected: %s",
              connInfo.getAddress().toString().c_str());
        LOG_I(deviceRole, "Connection interval: %.2f ms",
              connInfo.getConnInterval() * 1.25);

        applyConnectionParams(pServer, connInfo);

        onConnectHook(pServer, connInfo);
    }

    void onDisconnect(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) override
    {
        *connectionFlag = false;

        LOG_I(deviceRole, "Client disconnected (reason: %d)", reason);

        onDisconnectHook(pServer, connInfo, reason);
    }

    void onMTUChange(uint16_t MTU, NimBLEConnInfo &connInfo) override
    {
        LOG_I(deviceRole, "MTU updated: %u bytes", MTU);
    }

    uint32_t onPassKeyDisplay() override
    {
        uint32_t passkey = 123456;
        LOG_I(deviceRole, "PassKey Display: %06lu", (unsigned long)passkey);
        return passkey;
    }

    void onConfirmPassKey(NimBLEConnInfo &connInfo, uint32_t pin) override
    {
        LOG_I(deviceRole, "Confirm PIN: %06lu - auto-confirming", (unsigned long)pin);
        NimBLEDevice::injectConfirmPasskey(connInfo, true);
    }

    void onAuthenticationComplete(NimBLEConnInfo &connInfo) override
    {
        if (connInfo.isEncrypted())
        {
            LOG_I(deviceRole, "[OK] Authentication complete - ENCRYPTED");
        }
        else
        {
            LOG_W(deviceRole, "Authentication complete - NOT ENCRYPTED");
        }
    }
};

class BeaconServerCallbacks : public GenericServerCallbacks
{
public:
    BeaconServerCallbacks(std::atomic<bool> *connFlag, uint32_t *connCounter = nullptr)
        : GenericServerCallbacks("BEACON", connFlag, connCounter) {}

protected:
    void applyConnectionParams(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override
    {
        pServer->updateConnParams(
            connInfo.getConnHandle(),
            16,
            32,
            0,
            400);
    }
};

class AnchorServerCallbacks : public GenericServerCallbacks
{
private:
    std::atomic<bool> *scanningFlag;
    NimBLEScan *pScan;
    NimBLEAdvertising *pAdvertising;

public:
    AnchorServerCallbacks(std::atomic<bool> *connFlag, uint32_t *connCounter,
                          std::atomic<bool> *scanFlag, NimBLEScan *scan,
                          NimBLEAdvertising *adv = nullptr)
        : GenericServerCallbacks("ANCHOR", connFlag, connCounter),
          scanningFlag(scanFlag), pScan(scan), pAdvertising(adv) {}

protected:
    void applyConnectionParams(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override
    {
        pServer->updateConnParams(
            connInfo.getConnHandle(),
            8,
            24,
            0,
            400);
    }

    void onConnectHook(NimBLEServer *pServer, NimBLEConnInfo &connInfo) override
    {
        if (pServer->getConnectedCount() > 1)
        {
            LOG_W("ANCHOR", "Multiple clients detected! Rejecting extra connection.");
            LOG_W("ANCHOR", "   Current connections: %d (policy: max 1 client)",
                  pServer->getConnectedCount());
            LOG_W("ANCHOR", "   Rejecting: %s", connInfo.getAddress().toString().c_str());

            *connectionFlag = false;

            pServer->disconnect(connInfo.getConnHandle());

            return;
        }

        if (*scanningFlag && pScan != nullptr)
        {
            pScan->stop();
            *scanningFlag = false;
        }
    }

    void onDisconnectHook(NimBLEServer *pServer, NimBLEConnInfo &connInfo, int reason) override
    {
        if (pAdvertising != nullptr)
        {
            LOG_I("ANCHOR", "Restarting advertising after disconnect...");

            delay(100);

            if (pAdvertising->start())
            {
                LOG_I("ANCHOR", "[OK] Advertising restarted successfully");
            }
            else
            {
                LOG_E("ANCHOR", "Failed to restart advertising!");
            }
        }
    }
};
