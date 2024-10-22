#ifndef __XRD_STATISTICS_SERVICE_H_
#define __XRD_STATISTICS_SERVICE_H_

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/XrdAdaptor/interface/XrdStatistics.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;

  namespace service {
    class CondorStatusService;
  }
}  // namespace edm

namespace XrdAdaptor {

  class ClientRequest;
  class XrdReadStatistics;
  class XrdSiteStatistics;

  /* NOTE: All member information is kept in the XrdSiteStatisticsInformation singleton,
 * _not_ within the service itself.  This is because we need to be able to use the
 * singleton on non-CMSSW-created threads.  Services are only available to threads
 * created by CMSSW.
 */
  class XrdStatisticsService : public xrd_adaptor::XrdStatistics {
  public:
    XrdStatisticsService(const edm::ParameterSet &iPS, edm::ActivityRegistry &iRegistry);

    void postEndJob();

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

    // Provide an update of per-site transfer statistics to the CondorStatusService.
    // Returns a mapping of "site name" to transfer statistics.  The "site name" is
    // as self-identified by the Xrootd host; may not necessarily match up with the
    // "CMS site name".
    std::vector<std::pair<std::string, CondorIOStats>> condorUpdate() final;
  };

  class XrdSiteStatisticsInformation {
    friend class XrdStatisticsService;

  public:
    static XrdSiteStatisticsInformation *getInstance();

    std::shared_ptr<XrdSiteStatistics> getStatisticsForSite(std::string const &site);

  private:
    static void createInstance();

    static std::atomic<XrdSiteStatisticsInformation *> m_instance;
    std::mutex m_mutex;
    std::vector<edm::propagate_const<std::shared_ptr<XrdSiteStatistics>>> m_sites;
  };

  class XrdSiteStatistics {
    friend class XrdReadStatistics;

  public:
    XrdSiteStatistics(std::string const &site);
    XrdSiteStatistics(const XrdSiteStatistics &) = delete;
    XrdSiteStatistics &operator=(const XrdSiteStatistics &) = delete;

    std::string const &site() const { return m_site; }

    // Note that, while this function is thread-safe, the numbers are only consistent if no other
    // thread is reading data.
    void recomputeProperties(std::map<std::string, std::string> &props);

    static std::shared_ptr<XrdReadStatistics> startRead(std::shared_ptr<XrdSiteStatistics> parent,
                                                        std::shared_ptr<ClientRequest> req);

    void finishRead(XrdReadStatistics const &);

    uint64_t getTotalBytes() const { return m_readvSize + m_readSize; }
    std::chrono::nanoseconds getTotalReadTime() {
      return std::chrono::nanoseconds(m_readvNS) + std::chrono::nanoseconds(m_readNS);
    }

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

  class XrdReadStatistics {
    friend class XrdSiteStatistics;

  public:
    ~XrdReadStatistics() { m_parent->finishRead(*this); }
    XrdReadStatistics(const XrdReadStatistics &) = delete;
    XrdReadStatistics &operator=(const XrdReadStatistics &) = delete;

  private:
    XrdReadStatistics(std::shared_ptr<XrdSiteStatistics> parent, edm::storage::IOSize size, size_t count);

    uint64_t elapsedNS() const;
    int readCount() const { return m_count; }
    int size() const { return m_size; }

    size_t m_size;
    edm::storage::IOSize m_count;
    edm::propagate_const<std::shared_ptr<XrdSiteStatistics>> m_parent;
    std::chrono::time_point<std::chrono::steady_clock> m_start;
  };

}  // namespace XrdAdaptor

#endif
