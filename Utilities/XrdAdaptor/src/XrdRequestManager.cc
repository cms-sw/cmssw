
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>

#include <netdb.h>

#include "XrdCl/XrdClPostMasterInterfaces.hh"
#include "XrdCl/XrdClPostMaster.hh"

#include "XrdCl/XrdClFile.hh"
#include "XrdCl/XrdClDefaultEnv.hh"
#include "XrdCl/XrdClFileSystem.hh"

#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"

#include "XrdStatistics.h"
#include "Utilities/XrdAdaptor/src/XrdRequestManager.h"
#include "Utilities/XrdAdaptor/src/XrdHostHandler.hh"

static constexpr int XRD_CL_MAX_CHUNK = 512 * 1024;

static constexpr int XRD_ADAPTOR_SHORT_OPEN_DELAY = 5;

#ifdef XRD_FAKE_OPEN_PROBE
static constexpr int XRD_ADAPTOR_OPEN_PROBE_PERCENT = 100;
static constexpr int XRD_ADAPTOR_LONG_OPEN_DELAY = 20;
// This is the minimal difference in quality required to swap an active and inactive source
static constexpr int XRD_ADAPTOR_SOURCE_QUALITY_FUDGE = 0;
#else
static constexpr int XRD_ADAPTOR_OPEN_PROBE_PERCENT = 10;
static constexpr int XRD_ADAPTOR_LONG_OPEN_DELAY = 2 * 60;
static constexpr int XRD_ADAPTOR_SOURCE_QUALITY_FUDGE = 100;
#endif

static constexpr int XRD_ADAPTOR_CHUNK_THRESHOLD = 1000;

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#define GET_CLOCK_MONOTONIC(ts)                                      \
  {                                                                  \
    clock_serv_t cclock;                                             \
    mach_timespec_t mts;                                             \
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock); \
    clock_get_time(cclock, &mts);                                    \
    mach_port_deallocate(mach_task_self(), cclock);                  \
    ts.tv_sec = mts.tv_sec;                                          \
    ts.tv_nsec = mts.tv_nsec;                                        \
  }
#else
#define GET_CLOCK_MONOTONIC(ts) clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

using namespace XrdAdaptor;
using namespace edm::storage;

long long timeDiffMS(const timespec &a, const timespec &b) {
  long long diff = (a.tv_sec - b.tv_sec) * 1000;
  diff += (a.tv_nsec - b.tv_nsec) / 1e6;
  return diff;
}

/*
 * We do not care about the response of sending the monitoring information;
 * this handler class simply frees any returned buffer to prevent memory leaks.
 */
class SendMonitoringInfoHandler : public XrdCl::ResponseHandler {
  void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override {
    if (response) {
      XrdCl::Buffer *buffer = nullptr;
      response->Get(buffer);
      response->Set(static_cast<int *>(nullptr));
      delete buffer;
    }
    // Send Info has a response object; we must delete it.
    delete response;
    delete status;
    delete this;
  }

  XrdCl::FileSystem m_fs;

public:
  SendMonitoringInfoHandler(const SendMonitoringInfoHandler &) = delete;
  SendMonitoringInfoHandler &operator=(const SendMonitoringInfoHandler &) = delete;
  SendMonitoringInfoHandler() = delete;

  SendMonitoringInfoHandler(const std::string &url) : m_fs(url) {}

  XrdCl::FileSystem &fs() { return m_fs; }
};

static void SendMonitoringInfo(XrdCl::File &file) {
  // Do not send this to a dCache data server as they return an error.
  // In some versions of dCache, sending the monitoring information causes
  // the server to close the connection - resulting in failures.
  if (Source::isDCachePool(file)) {
    return;
  }

  // Send the monitoring info, if available.
  const char *jobId = edm::storage::StatisticsSenderService::getJobID();
  std::string lastUrl;
  file.GetProperty("LastURL", lastUrl);
  if (jobId && !lastUrl.empty()) {
    auto sm_handler = new SendMonitoringInfoHandler(lastUrl);
    if (!(sm_handler->fs().SendInfo(jobId, sm_handler, 30).IsOK())) {
      edm::LogWarning("XrdAdaptorInternal")
          << "Failed to send the monitoring information, monitoring ID is " << jobId << ".";
      delete sm_handler;
    }
    edm::LogInfo("XrdAdaptorInternal") << "Set monitoring ID to " << jobId << ".";
  }
}

namespace {
  std::unique_ptr<std::string> getQueryTransport(const XrdCl::URL &url, uint16_t query) {
    XrdCl::AnyObject result;
    XrdCl::DefaultEnv::GetPostMaster()->QueryTransport(url, query, result);
    std::string *tmp;
    result.Get(tmp);
    return std::unique_ptr<std::string>(tmp);
  }

  void tracerouteRedirections(const XrdCl::HostList *hostList) {
    edm::LogInfo("XrdAdaptorLvl2").log([hostList](auto &li) {
      int idx_redirection = 1;
      li << "-------------------------------\nTraceroute:\n";
      for (auto const &host : *hostList) {
        // Query info
        std::unique_ptr<std::string> stack_ip_method = getQueryTransport(host.url, XrdCl::StreamQuery::IpStack);
        std::unique_ptr<std::string> ip_method = getQueryTransport(host.url, XrdCl::StreamQuery::IpAddr);
        std::unique_ptr<std::string> auth_method = getQueryTransport(host.url, XrdCl::TransportQuery::Auth);
        std::unique_ptr<std::string> hostname_method = getQueryTransport(host.url, XrdCl::StreamQuery::HostName);
        std::string type_resource = "endpoint";
        std::string authentication;
        // Organize redirection info
        if (!auth_method->empty()) {
          authentication = *auth_method;
        } else {
          authentication = "no auth";
        };
        if (host.loadBalancer == 1) {
          type_resource = "load balancer";
        };
        li.format("{}. || {} / {} / {} / {} / {} / {} ||\n",
                  idx_redirection,
                  *hostname_method,
                  *stack_ip_method,
                  *ip_method,
                  host.url.GetPort(),
                  authentication,
                  type_resource);
        ++idx_redirection;
      }
      li.format("-------------------------------");
    });
  }
}  // namespace

RequestManager::RequestManager(const std::string &filename, XrdCl::OpenFlags::Flags flags, XrdCl::Access::Mode perms)
    : m_serverToAdvertise(nullptr),
      m_timeout(XRD_DEFAULT_TIMEOUT),
      m_nextInitialSourceToggle(false),
      m_redirectLimitDelayScale(1),
      m_name(filename),
      m_flags(flags),
      m_perms(perms),
      m_distribution(0, 100),
      m_excluded_active_count(0) {}

void RequestManager::initialize(std::weak_ptr<RequestManager> self) {
  m_open_handler = OpenHandler::getInstance(self);

  XrdCl::Env *env = XrdCl::DefaultEnv::GetEnv();
  if (env) {
    env->GetInt("StreamErrorWindow", m_timeout);
  }

  std::string orig_site;
  if (!Source::getXrootdSiteFromURL(m_name, orig_site) && (orig_site.find('.') == std::string::npos)) {
    std::string hostname;
    if (Source::getHostname(orig_site, hostname)) {
      Source::getDomain(hostname, orig_site);
    }
  }

  std::unique_ptr<XrdCl::File> file;
  edm::Exception ex(edm::errors::FileOpenError);
  bool validFile = false;
  const int retries = 5;
  std::string excludeString;
  for (int idx = 0; idx < retries; idx++) {
    file = std::make_unique<XrdCl::File>();
    auto opaque = prepareOpaqueString();
    std::string new_filename =
        m_name + (!opaque.empty() ? ((m_name.find('?') == m_name.npos) ? "?" : "&") + opaque : "");
    SyncHostResponseHandler handler;
    XrdCl::XRootDStatus openStatus = file->Open(new_filename, m_flags, m_perms, &handler);
    if (!openStatus
             .IsOK()) {  // In this case, we failed immediately - this indicates we have previously tried to talk to this
      // server and it was marked bad - xrootd couldn't even queue up the request internally!
      // In practice, we obsere this happening when the call to getXrootdSiteFromURL fails due to the
      // redirector being down or authentication failures.
      ex.clearMessage();
      ex.clearContext();
      ex.clearAdditionalInfo();
      ex << "XrdCl::File::Open(name='" << m_name << "', flags=0x" << std::hex << m_flags << ", permissions=0"
         << std::oct << m_perms << std::dec << ") => error '" << openStatus.ToStr() << "' (errno=" << openStatus.errNo
         << ", code=" << openStatus.code << ")";
      ex.addContext("Calling XrdFile::open()");
      ex.addAdditionalInfo("Remote server already encountered a fatal error; no redirections were performed.");
      throw ex;
    }
    handler.WaitForResponse();
    std::unique_ptr<XrdCl::XRootDStatus> status = handler.GetStatus();
    std::unique_ptr<XrdCl::HostList> hostList = handler.GetHosts();
    tracerouteRedirections(hostList.get());
    Source::determineHostExcludeString(*file, hostList.get(), excludeString);
    assert(status);
    if (status->IsOK()) {
      validFile = true;
      break;
    } else {
      ex.clearMessage();
      ex.clearContext();
      ex.clearAdditionalInfo();
      ex << "XrdCl::File::Open(name='" << m_name << "', flags=0x" << std::hex << m_flags << ", permissions=0"
         << std::oct << m_perms << std::dec << ") => error '" << status->ToStr() << "' (errno=" << status->errNo
         << ", code=" << status->code << ")";
      ex.addContext("Calling XrdFile::open()");
      addConnections(ex);
      std::string dataServer, lastUrl;
      file->GetProperty("DataServer", dataServer);
      file->GetProperty("LastURL", lastUrl);
      if (!dataServer.empty()) {
        ex.addAdditionalInfo("Problematic data server: " + dataServer);
      }
      if (!lastUrl.empty()) {
        ex.addAdditionalInfo("Last URL tried: " + lastUrl);
        edm::LogWarning("XrdAdaptorInternal") << "Failed to open file at URL " << lastUrl << ".";
      }
      if (std::find(m_disabledSourceStrings.begin(), m_disabledSourceStrings.end(), dataServer) !=
          m_disabledSourceStrings.end()) {
        ex << ". No additional data servers were found.";
        throw ex;
      }
      if (!dataServer.empty()) {
        m_disabledSourceStrings.insert(dataServer);
        m_disabledExcludeStrings.insert(excludeString);
      }
      // In this case, we didn't go anywhere - we stayed at the redirector and it gave us a file-not-found.
      if (lastUrl == new_filename) {
        edm::LogWarning("XrdAdaptorInternal") << lastUrl << ", " << new_filename;
        throw ex;
      }
    }
  }
  if (!validFile) {
    throw ex;
  }
  SendMonitoringInfo(*file);

  timespec ts;
  GET_CLOCK_MONOTONIC(ts);

  auto source = std::make_shared<Source>(ts, std::move(file), excludeString);
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    auto oldList = m_activeSources;
    m_activeSources.push_back(source);
    reportSiteChange(oldList, m_activeSources, orig_site);
  }
  queueUpdateCurrentServer(source->ID());
  updateCurrentServer();

  m_lastSourceCheck = ts;
  ts.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
  m_nextActiveSourceCheck = ts;
}

/**
 * Update the StatisticsSenderService with the current server info.
 *
 * As this accesses the edm::Service infrastructure, this MUST be called
 * from an edm-managed thread.  It CANNOT be called from an Xrootd-managed
 * thread.
 */
void RequestManager::updateCurrentServer() {
  // NOTE: we use memory_order_relaxed here, meaning that we may actually miss
  // a pending update.  *However*, since we call this for every read, we'll get it
  // eventually.
  if (LIKELY(!m_serverToAdvertise.load(std::memory_order_relaxed))) {
    return;
  }
  std::string *hostname_ptr;
  if ((hostname_ptr = m_serverToAdvertise.exchange(nullptr))) {
    std::unique_ptr<std::string> hostname(hostname_ptr);
    edm::Service<edm::storage::StatisticsSenderService> statsService;
    if (statsService.isAvailable()) {
      statsService->setCurrentServer(m_name, *hostname_ptr);
    }
  }
}

void RequestManager::queueUpdateCurrentServer(const std::string &id) {
  auto hostname = std::make_unique<std::string>(id);
  if (Source::getHostname(id, *hostname)) {
    std::string *null_hostname = nullptr;
    if (m_serverToAdvertise.compare_exchange_strong(null_hostname, hostname.get())) {
      hostname.release();
    }
  }
}

namespace {
  std::string formatSites(std::vector<std::shared_ptr<Source>> const &iSources) {
    std::string siteA, siteB;
    if (!iSources.empty()) {
      siteA = iSources[0]->Site();
    }
    if (iSources.size() == 2) {
      siteB = iSources[1]->Site();
    }
    std::string siteList = siteA;
    if (!siteB.empty() && (siteB != siteA)) {
      siteList = siteA + ", " + siteB;
    }
    return siteList;
  }
}  // namespace

void RequestManager::reportSiteChange(std::vector<std::shared_ptr<Source>> const &iOld,
                                      std::vector<std::shared_ptr<Source>> const &iNew,
                                      std::string orig_site) const {
  auto siteList = formatSites(iNew);
  if (orig_site.empty() || (orig_site == siteList)) {
    auto oldSites = formatSites(iOld);
  }

  edm::LogInfo("XrdAdaptorLvl1").log([&](auto &li) {
    li << "Serving data from: ";
    int size_active_sources = iNew.size();
    for (int i = 0; i < size_active_sources; ++i) {
      std::string hostname_a;
      Source::getHostname(iNew[i]->PrettyID(), hostname_a);
      li.format("   [{}] {}", i + 1, hostname_a);
    }
  });

  edm::LogInfo("XrdAdaptorLvl3").log([&](auto &li) {
    li << "The quality of the active sources is: ";
    int size_active_sources = iNew.size();
    for (int i = 0; i < size_active_sources; ++i) {
      std::string hostname_a;
      Source::getHostname(iNew[i]->PrettyID(), hostname_a);
      std::string quality = std::to_string(iNew[i]->getQuality());
      li.format("   [{}] {} for {}", i + 1, quality, hostname_a);
    }
  });
}

void RequestManager::checkSources(timespec &now,
                                  IOSize requestSize,
                                  std::vector<std::shared_ptr<Source>> &activeSources,
                                  std::vector<std::shared_ptr<Source>> &inactiveSources) {
  edm::LogVerbatim("XrdAdaptorInternal") << "Time since last check " << timeDiffMS(now, m_lastSourceCheck)
                                         << "; last check " << m_lastSourceCheck.tv_sec << "; now " << now.tv_sec
                                         << "; next check " << m_nextActiveSourceCheck.tv_sec << std::endl;
  if (timeDiffMS(now, m_lastSourceCheck) > 1000) {
    {  // Be more aggressive about getting rid of very bad sources.
      compareSources(now, 0, 1, activeSources, inactiveSources);
      compareSources(now, 1, 0, activeSources, inactiveSources);
    }
    if (timeDiffMS(now, m_nextActiveSourceCheck) > 0) {
      checkSourcesImpl(now, requestSize, activeSources, inactiveSources);
    }
  }
}

bool RequestManager::compareSources(const timespec &now,
                                    unsigned a,
                                    unsigned b,
                                    std::vector<std::shared_ptr<Source>> &activeSources,
                                    std::vector<std::shared_ptr<Source>> &inactiveSources) const {
  if (activeSources.size() < std::max(a, b) + 1) {
    return false;
  }
  unsigned quality_a = activeSources[a]->getQuality();
  unsigned quality_b = activeSources[b]->getQuality();
  bool findNewSource = false;
  if ((quality_a > 5130) || ((quality_a > 260) && (quality_b * 4 < quality_a))) {
    std::string hostname_a;
    Source::getHostname(activeSources[a]->ID(), hostname_a);
    if (quality_a > 5130) {
      edm::LogWarning("XrdAdaptorLvl3") << "Deactivating " << hostname_a << " from active sources because the quality ("
                                        << quality_a << ") is above 5130 and it is not the only active server";
    }
    if ((quality_a > 260) && (quality_b * 4 < quality_a)) {
      std::string hostname_b;
      Source::getHostname(activeSources[b]->ID(), hostname_b);
      edm::LogWarning("XrdAdaptorLvl3") << "Deactivating " << hostname_a << " from active sources because its quality ("
                                        << quality_a
                                        << ") is higher than 260 and 4 times larger than the other active server "
                                        << hostname_b << " (" << quality_b << ") ";
    }
    edm::LogVerbatim("XrdAdaptorInternal") << "Removing " << hostname_a << " from active sources due to poor quality ("
                                           << quality_a << " vs " << quality_b << ")" << std::endl;
    if (activeSources[a]->getLastDowngrade().tv_sec != 0) {
      findNewSource = true;
    }
    activeSources[a]->setLastDowngrade(now);
    inactiveSources.emplace_back(activeSources[a]);
    auto oldSources = activeSources;
    activeSources.erase(activeSources.begin() + a);
    reportSiteChange(oldSources, activeSources);
  }
  return findNewSource;
}

void RequestManager::checkSourcesImpl(timespec &now,
                                      IOSize requestSize,
                                      std::vector<std::shared_ptr<Source>> &activeSources,
                                      std::vector<std::shared_ptr<Source>> &inactiveSources) {
  bool findNewSource = false;
  if (activeSources.size() <= 1) {
    findNewSource = true;
    edm::LogInfo("XrdAdaptorLvl3")
        << "Looking for an additional source because the number of active sources is smaller than 2";
  } else if (activeSources.size() > 1) {
    edm::LogVerbatim("XrdAdaptorInternal") << "Source 0 quality " << activeSources[0]->getQuality()
                                           << ", source 1 quality " << activeSources[1]->getQuality() << std::endl;
    findNewSource |= compareSources(now, 0, 1, activeSources, inactiveSources);
    findNewSource |= compareSources(now, 1, 0, activeSources, inactiveSources);

    // NOTE: We could probably replace the copy with a better sort function.
    // However, there are typically very few sources and the correctness is more obvious right now.
    std::vector<std::shared_ptr<Source>> eligibleInactiveSources;
    eligibleInactiveSources.reserve(inactiveSources.size());
    for (const auto &source : inactiveSources) {
      if (timeDiffMS(now, source->getLastDowngrade()) > (XRD_ADAPTOR_SHORT_OPEN_DELAY - 1) * 1000) {
        eligibleInactiveSources.push_back(source);
      }
    }
    auto bestInactiveSource =
        std::min_element(eligibleInactiveSources.begin(),
                         eligibleInactiveSources.end(),
                         [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {
                           return s1->getQuality() < s2->getQuality();
                         });
    auto worstActiveSource = std::max_element(activeSources.cbegin(),
                                              activeSources.cend(),
                                              [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {
                                                return s1->getQuality() < s2->getQuality();
                                              });
    if (bestInactiveSource != eligibleInactiveSources.end() && bestInactiveSource->get()) {
      edm::LogVerbatim("XrdAdaptorInternal") << "Best inactive source: " << (*bestInactiveSource)->PrettyID()
                                             << ", quality " << (*bestInactiveSource)->getQuality();
    }
    edm::LogVerbatim("XrdAdaptorInternal") << "Worst active source: " << (*worstActiveSource)->PrettyID()
                                           << ", quality " << (*worstActiveSource)->getQuality();
    // Only upgrade the source if we only have one source and the best inactive one isn't too horrible.
    // Regardless, we will want to re-evaluate the new source quickly (within 5s).
    if ((bestInactiveSource != eligibleInactiveSources.end()) && activeSources.size() == 1 &&
        ((*bestInactiveSource)->getQuality() < 4 * activeSources[0]->getQuality())) {
      auto oldSources = activeSources;
      activeSources.push_back(*bestInactiveSource);
      reportSiteChange(oldSources, activeSources);
      for (auto it = inactiveSources.begin(); it != inactiveSources.end(); it++)
        if (it->get() == bestInactiveSource->get()) {
          inactiveSources.erase(it);
          break;
        }
    } else
      while ((bestInactiveSource != eligibleInactiveSources.end()) &&
             (*worstActiveSource)->getQuality() >
                 (*bestInactiveSource)->getQuality() + XRD_ADAPTOR_SOURCE_QUALITY_FUDGE) {
        edm::LogVerbatim("XrdAdaptorInternal")
            << "Removing " << (*worstActiveSource)->PrettyID() << " from active sources due to quality ("
            << (*worstActiveSource)->getQuality() << ") and promoting " << (*bestInactiveSource)->PrettyID()
            << " (quality: " << (*bestInactiveSource)->getQuality() << ")" << std::endl;
        (*worstActiveSource)->setLastDowngrade(now);
        for (auto it = inactiveSources.begin(); it != inactiveSources.end(); it++)
          if (it->get() == bestInactiveSource->get()) {
            inactiveSources.erase(it);
            break;
          }
        inactiveSources.emplace_back(*worstActiveSource);
        auto oldSources = activeSources;
        activeSources.erase(worstActiveSource);
        activeSources.emplace_back(std::move(*bestInactiveSource));
        reportSiteChange(oldSources, activeSources);
        eligibleInactiveSources.clear();
        for (const auto &source : inactiveSources)
          if (timeDiffMS(now, source->getLastDowngrade()) > (XRD_ADAPTOR_LONG_OPEN_DELAY - 1) * 1000)
            eligibleInactiveSources.push_back(source);
        bestInactiveSource = std::min_element(eligibleInactiveSources.begin(),
                                              eligibleInactiveSources.end(),
                                              [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {
                                                return s1->getQuality() < s2->getQuality();
                                              });
        worstActiveSource = std::max_element(activeSources.begin(),
                                             activeSources.end(),
                                             [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {
                                               return s1->getQuality() < s2->getQuality();
                                             });
      }
    if (!findNewSource && (timeDiffMS(now, m_lastSourceCheck) > 1000 * XRD_ADAPTOR_LONG_OPEN_DELAY)) {
      float r = m_distribution(m_generator);
      if (r < XRD_ADAPTOR_OPEN_PROBE_PERCENT) {
        findNewSource = true;
      }
    }
  }
  if (findNewSource) {
    m_open_handler->open();
    m_lastSourceCheck = now;
  }

  // Only aggressively look for new sources if we don't have two.
  if (activeSources.size() == 2) {
    now.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
  } else {
    now.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
  }
  m_nextActiveSourceCheck = now;
}

std::shared_ptr<XrdCl::File> RequestManager::getActiveFile() const {
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  if (m_activeSources.empty()) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdAdaptor::RequestManager::getActiveFile(name='" << m_name << "', flags=0x" << std::hex << m_flags
       << ", permissions=0" << std::oct << m_perms << std::dec << ") => Source used after fatal exception.";
    ex.addContext("In XrdAdaptor::RequestManager::handle()");
    addConnections(ex);
    throw ex;
  }
  return m_activeSources[0]->getFileHandle();
}

void RequestManager::getActiveSourceNames(std::vector<std::string> &sources) const {
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  sources.reserve(m_activeSources.size());
  for (auto const &source : m_activeSources) {
    sources.push_back(source->ID());
  }
}

void RequestManager::getPrettyActiveSourceNames(std::vector<std::string> &sources) const {
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  sources.reserve(m_activeSources.size());
  for (auto const &source : m_activeSources) {
    sources.push_back(source->PrettyID());
  }
}

void RequestManager::getDisabledSourceNames(std::vector<std::string> &sources) const {
  sources.reserve(m_disabledSourceStrings.size());
  for (auto const &source : m_disabledSourceStrings) {
    sources.push_back(source);
  }
}

void RequestManager::addConnections(cms::Exception &ex) const {
  std::vector<std::string> sources;
  getPrettyActiveSourceNames(sources);
  for (auto const &source : sources) {
    ex.addAdditionalInfo("Active source: " + source);
  }
  sources.clear();
  getDisabledSourceNames(sources);
  for (auto const &source : sources) {
    ex.addAdditionalInfo("Disabled source: " + source);
  }
}

std::shared_ptr<Source> RequestManager::pickSingleSource() {
  std::shared_ptr<Source> source = nullptr;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    if (m_activeSources.size() == 2) {
      if (m_nextInitialSourceToggle) {
        source = m_activeSources[0];
        m_nextInitialSourceToggle = false;
      } else {
        source = m_activeSources[1];
        m_nextInitialSourceToggle = true;
      }
    } else if (m_activeSources.empty()) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdAdaptor::RequestManager::handle read(name='" << m_name << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec << ") => Source used after fatal exception.";
      ex.addContext("In XrdAdaptor::RequestManager::handle()");
      addConnections(ex);
      throw ex;
    } else {
      source = m_activeSources[0];
    }
  }
  return source;
}

std::future<IOSize> RequestManager::handle(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr) {
  assert(c_ptr.get());
  timespec now;
  GET_CLOCK_MONOTONIC(now);
  //NOTE: can't hold lock while calling checkSources since can lead to lock inversion
  std::vector<std::shared_ptr<Source>> activeSources, inactiveSources;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    activeSources = m_activeSources;
    inactiveSources = m_inactiveSources;
  }
  {
    //make sure we update values before calling pickSingelSource
    std::shared_ptr<void *> guard(nullptr, [this, &activeSources, &inactiveSources](void *) {
      std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
      m_activeSources = std::move(activeSources);
      m_inactiveSources = std::move(inactiveSources);
    });

    checkSources(now, c_ptr->getSize(), activeSources, inactiveSources);
  }

  std::shared_ptr<Source> source = pickSingleSource();
  source->handle(c_ptr);
  return c_ptr->get_future();
}

std::string RequestManager::prepareOpaqueString() const {
  struct {
    std::stringstream ss;
    size_t count = 0;
    bool has_active = false;

    void append_tried(const std::string &id, bool active = false) {
      ss << (count ? "," : "tried=") << id;
      count++;
      if (active) {
        has_active = true;
      }
    }
  } state;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);

    for (const auto &it : m_activeSources) {
      state.append_tried(it->ExcludeID().substr(0, it->ExcludeID().find(':')), true);
    }
    for (const auto &it : m_inactiveSources) {
      state.append_tried(it->ExcludeID().substr(0, it->ExcludeID().find(':')));
    }
  }
  for (const auto &it : m_disabledExcludeStrings) {
    state.append_tried(it.substr(0, it.find(':')));
  }
  if (state.has_active) {
    state.ss << "&triedrc=resel";
  }

  return state.ss.str();
}

void XrdAdaptor::RequestManager::handleOpen(XrdCl::XRootDStatus &status, std::shared_ptr<Source> source) {
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  if (status.IsOK()) {
    edm::LogVerbatim("XrdAdaptorInternal") << "Successfully opened new source: " << source->PrettyID() << std::endl;
    m_redirectLimitDelayScale = 1;
    for (const auto &s : m_activeSources) {
      if (source->ID() == s->ID()) {
        edm::LogVerbatim("XrdAdaptorInternal")
            << "Xrootd server returned excluded source " << source->PrettyID() << "; ignoring" << std::endl;
        unsigned returned_count = ++m_excluded_active_count;
        m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
        if (returned_count >= 3) {
          m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - 2 * XRD_ADAPTOR_SHORT_OPEN_DELAY;
        }
        return;
      }
    }
    for (const auto &s : m_inactiveSources) {
      if (source->ID() == s->ID()) {
        edm::LogVerbatim("XrdAdaptorInternal")
            << "Xrootd server returned excluded inactive source " << source->PrettyID() << "; ignoring" << std::endl;
        m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
        return;
      }
    }
    if (m_activeSources.size() < 2) {
      auto oldSources = m_activeSources;
      m_activeSources.push_back(source);
      reportSiteChange(oldSources, m_activeSources);
      queueUpdateCurrentServer(source->ID());
    } else {
      m_inactiveSources.push_back(source);
    }
  } else {  // File-open failure - wait at least 120s before next attempt.
    edm::LogVerbatim("XrdAdaptorInternal") << "Got failure when trying to open a new source" << std::endl;
    int delayScale = 1;
    if (status.status == XrdCl::errRedirectLimit) {
      m_redirectLimitDelayScale = std::min(2 * m_redirectLimitDelayScale, 100);
      delayScale = m_redirectLimitDelayScale;
    }
    m_nextActiveSourceCheck.tv_sec += delayScale * XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
  }
}

std::future<IOSize> XrdAdaptor::RequestManager::handle(std::shared_ptr<std::vector<IOPosBuffer>> iolist) {
  //Use a copy of m_activeSources and m_inactiveSources throughout this function
  // in order to avoid holding the lock a long time and causing a deadlock.
  // When the function is over we will update the values of the containers
  std::vector<std::shared_ptr<Source>> activeSources, inactiveSources;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    activeSources = m_activeSources;
    inactiveSources = m_inactiveSources;
  }
  //Make sure we update changes when we leave the function
  std::shared_ptr<void *> guard(nullptr, [this, &activeSources, &inactiveSources](void *) {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    m_activeSources = std::move(activeSources);
    m_inactiveSources = std::move(inactiveSources);
  });

  updateCurrentServer();

  timespec now;
  GET_CLOCK_MONOTONIC(now);

  edm::CPUTimer timer;
  timer.start();

  if (activeSources.size() == 1) {
    auto c_ptr = std::make_shared<XrdAdaptor::ClientRequest>(*this, iolist);
    checkSources(now, c_ptr->getSize(), activeSources, inactiveSources);
    activeSources[0]->handle(c_ptr);
    return c_ptr->get_future();
  }
  // Make sure active
  else if (activeSources.empty()) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdAdaptor::RequestManager::handle readv(name='" << m_name << "', flags=0x" << std::hex << m_flags
       << ", permissions=0" << std::oct << m_perms << std::dec << ") => Source used after fatal exception.";
    ex.addContext("In XrdAdaptor::RequestManager::handle()");
    addConnections(ex);
    throw ex;
  }

  assert(iolist.get());
  auto req1 = std::make_shared<std::vector<IOPosBuffer>>();
  auto req2 = std::make_shared<std::vector<IOPosBuffer>>();
  splitClientRequest(*iolist, *req1, *req2, activeSources);

  checkSources(now, req1->size() + req2->size(), activeSources, inactiveSources);
  // CheckSources may have removed a source
  if (activeSources.size() == 1) {
    auto c_ptr = std::make_shared<XrdAdaptor::ClientRequest>(*this, iolist);
    activeSources[0]->handle(c_ptr);
    return c_ptr->get_future();
  }

  std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr1, c_ptr2;
  std::future<IOSize> future1, future2;
  if (!req1->empty()) {
    c_ptr1.reset(new XrdAdaptor::ClientRequest(*this, req1));
    activeSources[0]->handle(c_ptr1);
    future1 = c_ptr1->get_future();
  }
  if (!req2->empty()) {
    c_ptr2.reset(new XrdAdaptor::ClientRequest(*this, req2));
    activeSources[1]->handle(c_ptr2);
    future2 = c_ptr2->get_future();
  }
  if (!req1->empty() && !req2->empty()) {
    std::future<IOSize> task = std::async(
        std::launch::deferred,
        [](std::future<IOSize> a, std::future<IOSize> b) {
          // Wait until *both* results are available.  This is essential
          // as the callback may try referencing the RequestManager.  If one
          // throws an exception (causing the RequestManager to be destroyed by
          // XrdFile) and the other has a failure, then the recovery code will
          // reference the destroyed RequestManager.
          //
          // Unlike other places where we use shared/weak ptrs to maintain object
          // lifetime and destruction asynchronously, we *cannot* destroy the request
          // asynchronously as it is associated with a ROOT buffer.  We must wait until we
          // are guaranteed that XrdCl will not write into the ROOT buffer before we
          // can return.
          b.wait();
          a.wait();
          return b.get() + a.get();
        },
        std::move(future1),
        std::move(future2));
    timer.stop();
    //edm::LogVerbatim("XrdAdaptorInternal") << "Total time to create requests " << static_cast<int>(1000*timer.realTime()) << std::endl;
    return task;
  } else if (!req1->empty()) {
    return future1;
  } else if (!req2->empty()) {
    return future2;
  } else {  // Degenerate case - no bytes to read.
    std::promise<IOSize> p;
    p.set_value(0);
    return p.get_future();
  }
}

void RequestManager::requestFailure(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr, XrdCl::Status &c_status) {
  std::shared_ptr<Source> source_ptr = c_ptr->getCurrentSource();

  // Fail early for invalid responses - XrdFile has a separate path for handling this.
  if (c_status.code == XrdCl::errInvalidResponse) {
    edm::LogWarning("XrdAdaptorInternal") << "Invalid response when reading from " << source_ptr->PrettyID();
    XrootdException ex(c_status, edm::errors::FileReadError);
    ex << "XrdAdaptor::RequestManager::requestFailure readv(name='" << m_name << "', flags=0x" << std::hex << m_flags
       << ", permissions=0" << std::oct << m_perms << std::dec << ", old source=" << source_ptr->PrettyID()
       << ") => Invalid ReadV response from server";
    ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
    addConnections(ex);
    throw ex;
  }
  edm::LogWarning("XrdAdaptorInternal") << "Request failure when reading from " << source_ptr->PrettyID();

  // Note that we do not delete the Source itself.  That is because this
  // function may be called from within XrdCl::ResponseHandler::HandleResponseWithHosts
  // In such a case, if you close a file in the handler, it will deadlock
  m_disabledSourceStrings.insert(source_ptr->ID());
  m_disabledExcludeStrings.insert(source_ptr->ExcludeID());
  m_disabledSources.insert(source_ptr);

  std::unique_lock<std::recursive_mutex> sentry(m_source_mutex);
  if ((!m_activeSources.empty()) && (m_activeSources[0].get() == source_ptr.get())) {
    auto oldSources = m_activeSources;
    m_activeSources.erase(m_activeSources.begin());
    reportSiteChange(oldSources, m_activeSources);
  } else if ((m_activeSources.size() > 1) && (m_activeSources[1].get() == source_ptr.get())) {
    auto oldSources = m_activeSources;
    m_activeSources.erase(m_activeSources.begin() + 1);
    reportSiteChange(oldSources, m_activeSources);
  }
  std::shared_ptr<Source> new_source;
  if (m_activeSources.empty()) {
    std::shared_future<std::shared_ptr<Source>> future = m_open_handler->open();
    timespec now;
    GET_CLOCK_MONOTONIC(now);
    m_lastSourceCheck = now;
    // Note we only wait for 180 seconds here.  This is because we've already failed
    // once and the likelihood the program has some inconsistent state is decent.
    // We'd much rather fail hard than deadlock!
    sentry.unlock();
    std::future_status status = future.wait_for(std::chrono::seconds(m_timeout + 10));
    if (status == std::future_status::timeout) {
      XrootdException ex(c_status, edm::errors::FileOpenError);
      ex << "XrdAdaptor::RequestManager::requestFailure Open(name='" << m_name << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec << ", old source=" << source_ptr->PrettyID()
         << ") => timeout when waiting for file open";
      ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
      addConnections(ex);
      throw ex;
    } else {
      try {
        new_source = future.get();
      } catch (edm::Exception &ex) {
        ex.addContext("Handling XrdAdaptor::RequestManager::requestFailure()");
        ex.addAdditionalInfo("Original failed source is " + source_ptr->PrettyID());
        throw;
      }
    }

    if (std::find(m_disabledSourceStrings.begin(), m_disabledSourceStrings.end(), new_source->ID()) !=
        m_disabledSourceStrings.end()) {
      // The server gave us back a data node we requested excluded.  Fatal!
      XrootdException ex(c_status, edm::errors::FileOpenError);
      ex << "XrdAdaptor::RequestManager::requestFailure Open(name='" << m_name << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec << ", old source=" << source_ptr->PrettyID()
         << ", new source=" << new_source->PrettyID() << ") => Xrootd server returned an excluded source";
      ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
      addConnections(ex);
      throw ex;
    }
    sentry.lock();

    auto oldSources = m_activeSources;
    m_activeSources.push_back(new_source);
    reportSiteChange(oldSources, m_activeSources);
  } else {
    new_source = m_activeSources[0];
  }
  new_source->handle(c_ptr);
}

static void consumeChunkFront(size_t &front,
                              std::vector<IOPosBuffer> &input,
                              std::vector<IOPosBuffer> &output,
                              IOSize chunksize) {
  while ((chunksize > 0) && (front < input.size()) && (output.size() <= XRD_ADAPTOR_CHUNK_THRESHOLD)) {
    IOPosBuffer &io = input[front];
    IOPosBuffer &outio = output.back();
    if (io.size() > chunksize) {
      IOSize consumed;
      if (!output.empty() && (outio.size() < XRD_CL_MAX_CHUNK) &&
          (outio.offset() + static_cast<IOOffset>(outio.size()) == io.offset())) {
        if (outio.size() + chunksize > XRD_CL_MAX_CHUNK) {
          consumed = (XRD_CL_MAX_CHUNK - outio.size());
          outio.set_size(XRD_CL_MAX_CHUNK);
        } else {
          consumed = chunksize;
          outio.set_size(outio.size() + consumed);
        }
      } else {
        consumed = chunksize;
        output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
      }
      chunksize -= consumed;
      IOSize newsize = io.size() - consumed;
      IOOffset newoffset = io.offset() + consumed;
      void *newdata = static_cast<char *>(io.data()) + consumed;
      io.set_offset(newoffset);
      io.set_data(newdata);
      io.set_size(newsize);
    } else if (io.size() == 0) {
      front++;
    } else {
      output.push_back(io);
      chunksize -= io.size();
      front++;
    }
  }
}

static void consumeChunkBack(size_t front,
                             std::vector<IOPosBuffer> &input,
                             std::vector<IOPosBuffer> &output,
                             IOSize chunksize) {
  while ((chunksize > 0) && (front < input.size()) && (output.size() <= XRD_ADAPTOR_CHUNK_THRESHOLD)) {
    IOPosBuffer &io = input.back();
    IOPosBuffer &outio = output.back();
    if (io.size() > chunksize) {
      IOSize consumed;
      if (!output.empty() && (outio.size() < XRD_CL_MAX_CHUNK) &&
          (outio.offset() + static_cast<IOOffset>(outio.size()) == io.offset())) {
        if (outio.size() + chunksize > XRD_CL_MAX_CHUNK) {
          consumed = (XRD_CL_MAX_CHUNK - outio.size());
          outio.set_size(XRD_CL_MAX_CHUNK);
        } else {
          consumed = chunksize;
          outio.set_size(outio.size() + consumed);
        }
      } else {
        consumed = chunksize;
        output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
      }
      chunksize -= consumed;
      IOSize newsize = io.size() - consumed;
      IOOffset newoffset = io.offset() + consumed;
      void *newdata = static_cast<char *>(io.data()) + consumed;
      io.set_offset(newoffset);
      io.set_data(newdata);
      io.set_size(newsize);
    } else if (io.size() == 0) {
      input.pop_back();
    } else {
      output.push_back(io);
      chunksize -= io.size();
      input.pop_back();
    }
  }
}

static IOSize validateList(const std::vector<IOPosBuffer> req) {
  IOSize total = 0;
  off_t last_offset = -1;
  for (const auto &it : req) {
    total += it.size();
    assert(it.offset() > last_offset);
    last_offset = it.offset();
    assert(it.size() <= XRD_CL_MAX_CHUNK);
    assert(it.offset() < 0x1ffffffffff);
  }
  assert(req.size() <= 1024);
  return total;
}

void XrdAdaptor::RequestManager::splitClientRequest(const std::vector<IOPosBuffer> &iolist,
                                                    std::vector<IOPosBuffer> &req1,
                                                    std::vector<IOPosBuffer> &req2,
                                                    std::vector<std::shared_ptr<Source>> const &activeSources) const {
  if (iolist.empty())
    return;
  std::vector<IOPosBuffer> tmp_iolist(iolist.begin(), iolist.end());
  req1.reserve(iolist.size() / 2 + 1);
  req2.reserve(iolist.size() / 2 + 1);
  size_t front = 0;

  // The quality of both is increased by 5 to prevent strange effects if quality is 0 for one source.
  float q1 = static_cast<float>(activeSources[0]->getQuality()) + 5;
  float q2 = static_cast<float>(activeSources[1]->getQuality()) + 5;
  IOSize chunk1, chunk2;
  // Make sure the chunk size is at least 1024; little point to reads less than that size.
  chunk1 = std::max(static_cast<IOSize>(static_cast<float>(XRD_CL_MAX_CHUNK) * (q2 * q2 / (q1 * q1 + q2 * q2))),
                    static_cast<IOSize>(1024));
  chunk2 = std::max(static_cast<IOSize>(static_cast<float>(XRD_CL_MAX_CHUNK) * (q1 * q1 / (q1 * q1 + q2 * q2))),
                    static_cast<IOSize>(1024));

  IOSize size_orig = 0;
  for (const auto &it : iolist)
    size_orig += it.size();

  while (tmp_iolist.size() - front > 0) {
    if ((req1.size() >= XRD_ADAPTOR_CHUNK_THRESHOLD) &&
        (req2.size() >=
         XRD_ADAPTOR_CHUNK_THRESHOLD)) {  // The XrdFile::readv implementation should guarantee that no more than approximately 1024 chunks
      // are passed to the request manager.  However, because we have a max chunk size, we increase
      // the total number slightly.  Theoretically, it's possible an individual readv of total size >2GB where
      // each individual chunk is >1MB could result in this firing.  However, within the context of CMSSW,
      // this cannot happen (ROOT uses readv for TTreeCache; TTreeCache size is 20MB).
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdAdaptor::RequestManager::splitClientRequest(name='" << m_name << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec
         << ") => Unable to split request between active servers.  This is an unexpected internal error and should be "
            "reported to CMSSW developers.";
      ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
      addConnections(ex);
      std::stringstream ss;
      ss << "Original request size " << iolist.size() << "(" << size_orig << " bytes)";
      ex.addAdditionalInfo(ss.str());
      std::stringstream ss2;
      ss2 << "Quality source 1 " << q1 - 5 << ", quality source 2: " << q2 - 5;
      ex.addAdditionalInfo(ss2.str());
      throw ex;
    }
    if (req1.size() < XRD_ADAPTOR_CHUNK_THRESHOLD) {
      consumeChunkFront(front, tmp_iolist, req1, chunk1);
    }
    if (req2.size() < XRD_ADAPTOR_CHUNK_THRESHOLD) {
      consumeChunkBack(front, tmp_iolist, req2, chunk2);
    }
  }
  std::sort(req1.begin(), req1.end(), [](const IOPosBuffer &left, const IOPosBuffer &right) {
    return left.offset() < right.offset();
  });
  std::sort(req2.begin(), req2.end(), [](const IOPosBuffer &left, const IOPosBuffer &right) {
    return left.offset() < right.offset();
  });

  IOSize size1 = validateList(req1);
  IOSize size2 = validateList(req2);

  assert(size_orig == size1 + size2);

  edm::LogVerbatim("XrdAdaptorInternal") << "Original request size " << iolist.size() << " (" << size_orig
                                         << " bytes) split into requests size " << req1.size() << " (" << size1
                                         << " bytes) and " << req2.size() << " (" << size2 << " bytes)" << std::endl;
}

XrdAdaptor::RequestManager::OpenHandler::OpenHandler(std::weak_ptr<RequestManager> manager) : m_manager(manager) {}

// Cannot use ~OpenHandler=default as XrdCl::File is not fully
// defined in the header.
XrdAdaptor::RequestManager::OpenHandler::~OpenHandler() {}

void XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts(XrdCl::XRootDStatus *status_ptr,
                                                                      XrdCl::AnyObject *,
                                                                      XrdCl::HostList *hostList_ptr) {
  // Make sure we get rid of the strong self-reference when the callback finishes.
  std::shared_ptr<OpenHandler> self = m_self;
  m_self.reset();

  // NOTE: as in XrdCl::File (synchronous), we ignore the response object.
  // Make sure that we set m_outstanding_open to false on exit from this function.
  // NOTE: we need to pass non-nullptr to unique_ptr in order for the guard to run
  std::unique_ptr<OpenHandler, std::function<void(OpenHandler *)>> outstanding_guard(
      this, [&](OpenHandler *) { m_outstanding_open = false; });

  std::shared_ptr<Source> source;
  std::unique_ptr<XrdCl::XRootDStatus> status(status_ptr);
  std::unique_ptr<XrdCl::HostList> hostList(hostList_ptr);
  tracerouteRedirections(hostList.get());
  auto manager = m_manager.lock();
  // Manager object has already been deleted.  Cleanup the
  // response objects, remove our self-reference, and ignore the response.
  if (!manager) {
    return;
  }
  //if we need to delete the File object we must do it outside
  // of the lock to avoid a potential deadlock
  std::unique_ptr<XrdCl::File> releaseFile;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);

    if (status->IsOK()) {
      SendMonitoringInfo(*m_file);
      timespec now;
      GET_CLOCK_MONOTONIC(now);

      std::string excludeString;
      Source::determineHostExcludeString(*m_file, hostList.get(), excludeString);

      source.reset(new Source(now, std::move(m_file), excludeString));
      m_promise.set_value(source);
    } else {
      releaseFile = std::move(m_file);
      edm::Exception ex(edm::errors::FileOpenError);
      ex << "XrdCl::File::Open(name='" << manager->m_name << "', flags=0x" << std::hex << manager->m_flags
         << ", permissions=0" << std::oct << manager->m_perms << std::dec << ") => error '" << status->ToStr()
         << "' (errno=" << status->errNo << ", code=" << status->code << ")";
      ex.addContext("In XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts()");
      manager->addConnections(ex);
      m_promise.set_exception(std::make_exception_ptr(ex));
    }
  }
  manager->handleOpen(*status, source);
}

std::string XrdAdaptor::RequestManager::OpenHandler::current_source() {
  std::lock_guard<std::recursive_mutex> sentry(m_mutex);

  if (!m_file.get()) {
    return "(no open in progress)";
  }
  std::string dataServer;
  m_file->GetProperty("DataServer", dataServer);
  if (dataServer.empty()) {
    return "(unknown source)";
  }
  return dataServer;
}

std::shared_future<std::shared_ptr<Source>> XrdAdaptor::RequestManager::OpenHandler::open() {
  auto manager_ptr = m_manager.lock();
  if (!manager_ptr) {
    edm::Exception ex(edm::errors::LogicError);
    ex << "XrdCl::File::Open() =>"
       << " error: OpenHandler called within an invalid RequestManager context."
       << "  This is a logic error and should be reported to the CMSSW developers.";
    ex.addContext("Calling XrdAdaptor::RequestManager::OpenHandler::open()");
    throw ex;
  }
  RequestManager &manager = *manager_ptr;
  auto self_ptr = m_self_weak.lock();
  if (!self_ptr) {
    edm::Exception ex(edm::errors::LogicError);
    ex << "XrdCl::File::Open() => error: "
       << "OpenHandler called after it was deleted.  This is a logic error "
       << "and should be reported to the CMSSW developers.";
    ex.addContext("Calling XrdAdapter::RequestManager::OpenHandler::open()");
    throw ex;
  }

  // NOTE NOTE: we look at this variable *without* the lock.  This means the method
  // is not thread-safe; the caller is responsible to verify it is not called from
  // multiple threads simultaneously.
  //
  // This is done because ::open may be called from a Xrootd callback; if we
  // tried to hold m_mutex here, this object's callback may also be active, hold m_mutex,
  // and make a call into xrootd (when it invokes m_file.reset()).  Hence, our callback
  // holds our mutex and attempts to grab an Xrootd mutex; RequestManager::requestFailure holds
  // an Xrootd mutex and tries to hold m_mutex.  This is a classic deadlock.
  if (m_outstanding_open) {
    return m_shared_future;
  }
  std::lock_guard<std::recursive_mutex> sentry(m_mutex);
  std::promise<std::shared_ptr<Source>> new_promise;
  m_promise.swap(new_promise);
  m_shared_future = m_promise.get_future().share();

  auto opaque = manager.prepareOpaqueString();
  std::string new_name = manager.m_name + ((manager.m_name.find('?') == manager.m_name.npos) ? "?" : "&") + opaque;
  edm::LogVerbatim("XrdAdaptorInternal") << "Trying to open URL: " << new_name;
  m_file = std::make_unique<XrdCl::File>();
  m_outstanding_open = true;

  // Always make sure we release m_file and set m_outstanding_open to false on error.
  std::unique_ptr<OpenHandler, std::function<void(OpenHandler *)>> exit_guard(this, [&](OpenHandler *) {
    m_outstanding_open = false;
    m_file.reset();
  });

  XrdCl::XRootDStatus status;
  if (!(status = m_file->Open(new_name, manager.m_flags, manager.m_perms, this)).IsOK()) {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdCl::File::Open(name='" << new_name << "', flags=0x" << std::hex << manager.m_flags << ", permissions=0"
       << std::oct << manager.m_perms << std::dec << ") => error '" << status.ToStr() << "' (errno=" << status.errNo
       << ", code=" << status.code << ")";
    ex.addContext("Calling XrdAdaptor::RequestManager::OpenHandler::open()");
    manager.addConnections(ex);
    throw ex;
  }
  exit_guard.release();
  // Have a strong self-reference for as long as the callback is in-progress.
  m_self = self_ptr;
  return m_shared_future;
}
