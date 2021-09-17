#ifndef Utilities_XrdAdaptor_XrdSource_h
#define Utilities_XrdAdaptor_XrdSource_h

#include "XrdCl/XrdClXRootDResponses.hh"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>
#include <vector>

#include "QualityMetric.h"

namespace XrdCl {
  class File;
}

namespace XrdAdaptor {

  class RequestList;
  class ClientRequest;
  class XrdSiteStatistics;
  class XrdStatisticsService;

  class Source : public std::enable_shared_from_this<Source> {
  public:
    Source(const Source &) = delete;
    Source &operator=(const Source &) = delete;

    Source(timespec now, std::unique_ptr<XrdCl::File> fileHandle, const std::string &exclude);

    ~Source();

    void handle(std::shared_ptr<ClientRequest>);

    void handle(RequestList &);

    std::shared_ptr<XrdCl::File> getFileHandle();

    const std::string &ID() const { return m_id; }
    const std::string &Site() const { return m_site; }
    const std::string &PrettyID() const { return m_prettyid; }
    const std::string &ExcludeID() const { return m_exclude; }

    unsigned getQuality() { return m_qm->get(); }

    struct timespec getLastDowngrade() const {
      return m_lastDowngrade;
    }
    void setLastDowngrade(struct timespec now) { m_lastDowngrade = now; }

    static bool getDomain(const std::string &host, std::string &domain);
    static bool getXrootdSite(XrdCl::File &file, std::string &site);
    static bool getXrootdSiteFromURL(std::string url, std::string &site);

    // Given a file and (possibly) a host list, determine the exclude string.
    static void determineHostExcludeString(XrdCl::File &file, const XrdCl::HostList *hostList, std::string &exclude);

    // Given a connected File object, determine whether we believe this to be a
    // dCache pool (dCache is a separate implementation and sometimes benefits from
    // implementation-specific behaviors.
    static bool isDCachePool(XrdCl::File &file, const XrdCl::HostList *hostList = nullptr);
    static bool isDCachePool(const std::string &url);

    // Given an Xrootd server ID, determine the hostname to the best of our ability.
    static bool getHostname(const std::string &id, std::string &hostname);

  private:
    void requestCallback(/* TODO: type? */);

    void setXrootdSite();

    std::shared_ptr<XrdCl::File const> fh() const { return get_underlying_safe(m_fh); }
    std::shared_ptr<XrdCl::File> &fh() { return get_underlying_safe(m_fh); }
    std::shared_ptr<XrdSiteStatistics const> stats() const { return get_underlying_safe(m_stats); }
    std::shared_ptr<XrdSiteStatistics> &stats() { return get_underlying_safe(m_stats); }

    struct timespec m_lastDowngrade;
    std::string m_id;
    std::string m_prettyid;
    std::string m_site;
    std::string m_exclude;
    edm::propagate_const<std::shared_ptr<XrdCl::File>> m_fh;

    edm::propagate_const<std::unique_ptr<QualityMetricSource>> m_qm;
    edm::propagate_const<std::shared_ptr<XrdSiteStatistics>> m_stats;

#ifdef XRD_FAKE_SLOW
    bool m_slow;
#endif
  };

}  // namespace XrdAdaptor

#endif
