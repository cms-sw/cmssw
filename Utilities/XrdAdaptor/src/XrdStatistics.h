#ifndef __XRD_STATISTICS_SERVICE_H_
#define __XRD_STATISTICS_SERVICE_H_

#include "Utilities/StorageFactory/interface/IOTypes.h"

#include <chrono>
#include <mutex>
#include <atomic>

namespace edm
{
    class ParameterSet;
    class ActivityRegistry;
    class ConfigurationDescriptions;
}

namespace XrdAdaptor
{

class ClientRequest;
class XrdReadStatistics;
class XrdSiteStatistics;


/* NOTE: All member information is kept in the XrdSiteStatisticsInformation singleton,
 * _not_ within the service itself.  This is because we need to be able to use the
 * singleton on non-CMSSW-created threads.  Services are only available to threads
 * created by CMSSW.
 */
class XrdStatisticsService
{
public:

    XrdStatisticsService(const edm::ParameterSet& iPS, edm::ActivityRegistry &iRegistry);

    void postEndJob();

    void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:

};

class XrdSiteStatisticsInformation
{
friend class XrdStatisticsService;

public:
    static XrdSiteStatisticsInformation *getInstance();

    std::shared_ptr<XrdSiteStatistics> getStatisticsForSite(std::string const &site);

private:
    static void createInstance();

    static std::atomic<XrdSiteStatisticsInformation*> m_instance;
    std::mutex m_mutex;
    std::vector<std::shared_ptr<XrdSiteStatistics>> m_sites;
};

class XrdSiteStatistics
{
friend class XrdReadStatistics;

public:
    XrdSiteStatistics(std::string const &site);
    XrdSiteStatistics(const XrdSiteStatistics&) = delete;
    XrdSiteStatistics &operator=(const XrdSiteStatistics&) = delete;

    std::string const &site() const {return m_site;}

    // Note that, while this function is thread-safe, the numbers are only consistent if no other
    // thread is reading data.
    void recomputeProperties(std::map<std::string, std::string> &props);

    static std::shared_ptr<XrdReadStatistics> startRead(std::shared_ptr<XrdSiteStatistics> parent, std::shared_ptr<ClientRequest> req);

    void finishRead(XrdReadStatistics const &);

private:
    const std::string m_site = "Unknown";

    std::atomic<unsigned> m_readvCount;
    std::atomic<unsigned> m_chunkCount;
    std::atomic<uint64_t> m_readvSize;
    std::atomic<uint64_t> m_readvNS;
    std::atomic<unsigned> m_readCount;
    std::atomic<uint64_t> m_readSize;
    std::atomic<uint64_t> m_readNS;
};

class XrdReadStatistics
{
friend class XrdSiteStatistics;

public:
    ~XrdReadStatistics() {m_parent->finishRead(*this);}
    XrdReadStatistics(const XrdReadStatistics&) = delete;
    XrdReadStatistics &operator=(const XrdReadStatistics&) = delete;

private:
    XrdReadStatistics(std::shared_ptr<XrdSiteStatistics> parent, IOSize size, size_t count);

    float elapsedNS() const;
    int readCount() const {return m_count;}
    int size() const {return m_size;}

    size_t m_size;
    IOSize m_count;
    std::shared_ptr<XrdSiteStatistics> m_parent;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

}

#endif
