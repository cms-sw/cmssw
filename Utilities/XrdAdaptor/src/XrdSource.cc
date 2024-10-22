
// See http://stackoverflow.com/questions/12523122/what-is-glibcxx-use-nanosleep-all-about
#define _GLIBCXX_USE_NANOSLEEP
#include <memory>

#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <cassert>
#include <netdb.h>

#include "XrdCl/XrdClFile.hh"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "XrdSource.h"
#include "XrdRequest.h"
#include "QualityMetric.h"
#include "XrdStatistics.h"

#define MAX_REQUEST 256 * 1024
#define XRD_CL_MAX_CHUNK 512 * 1024

#ifdef XRD_FAKE_SLOW
//#define XRD_DELAY 5140
#define XRD_DELAY 1000
#define XRD_SLOW_RATE 2
std::atomic<int> g_delayCount{0};
#else
std::atomic<int> g_delayCount{0};
#endif

using namespace XrdAdaptor;

// File::Close() can take awhile - slow servers (which are probably
// inactive anyway!) can even timeout.  Rather than wait around for
// a few minutes in the main thread, this class asynchronously closes
// and deletes the XrdCl::File
class DelayedClose : public XrdCl::ResponseHandler {
public:
  DelayedClose(const DelayedClose &) = delete;
  DelayedClose &operator=(const DelayedClose &) = delete;

  DelayedClose(std::shared_ptr<XrdCl::File> fh, const std::string &id, const std::string &site)
      : m_fh(std::move(fh)), m_id(id), m_site(site) {
    if (m_fh && m_fh->IsOpen()) {
      if (!m_fh->Close(this).IsOK()) {
        delete this;
      }
    }
  }

  ~DelayedClose() override = default;

  void HandleResponseWithHosts(XrdCl::XRootDStatus *status,
                               XrdCl::AnyObject *response,
                               XrdCl::HostList *hostList) override {
    if (status && !status->IsOK()) {
      edm::LogWarning("XrdFileWarning") << "Source delayed close failed with error '" << status->ToStr()
                                        << "' (errno=" << status->errNo << ", code=" << status->code
                                        << ", server=" << m_id << ", site=" << m_site << ")";
    }
    delete status;
    delete hostList;
    // NOTE: we do not delete response (copying behavior from XrdCl).
    delete this;
  }

private:
  edm::propagate_const<std::shared_ptr<XrdCl::File>> m_fh;
  std::string m_id;
  std::string m_site;
};

/**
 * A handler for querying a XrdCl::FileSystem object which is safe to be
 * invoked from an XrdCl callback (that is, we don't need an available callback
 * thread to timeout).
 */
class QueryAttrHandler : public XrdCl::ResponseHandler {
  friend std::unique_ptr<QueryAttrHandler> std::make_unique<QueryAttrHandler>();

public:
  QueryAttrHandler() = delete;
  ~QueryAttrHandler() override = default;
  QueryAttrHandler(const QueryAttrHandler &) = delete;
  QueryAttrHandler &operator=(const QueryAttrHandler &) = delete;

  QueryAttrHandler(const std::string &url) : m_fs(url) {}

  static XrdCl::XRootDStatus query(const std::string &url,
                                   const std::string &attr,
                                   std::chrono::milliseconds timeout,
                                   std::string &result) {
    auto handler = std::make_unique<QueryAttrHandler>(url);
    auto l_state = std::make_shared<QueryAttrState>();
    handler->m_state = l_state;
    XrdCl::Buffer arg(attr.size());
    arg.FromString(attr);

    XrdCl::XRootDStatus st = handler->m_fs.Query(XrdCl::QueryCode::Config, arg, handler.get());
    if (!st.IsOK()) {
      return st;
    }

    // Successfully registered the callback; it will always delete itself, so we shouldn't.
    handler.release();

    std::unique_lock<std::mutex> guard(l_state->m_mutex);
    // Wait until some status is available or a timeout.
    l_state->m_condvar.wait_for(guard, timeout, [&] { return l_state->m_status.get(); });

    if (l_state->m_status) {
      if (l_state->m_status->IsOK()) {
        result = l_state->m_response->ToString();
      }
      return *(l_state->m_status);
    } else {  // We had a timeout; construct a reasonable message.
      return XrdCl::XRootDStatus(
          XrdCl::stError, XrdCl::errSocketTimeout, 1, "Timeout when waiting for query callback.");
    }
  }

private:
  void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override {
    // NOTE: we own the status and response pointers.
    std::unique_ptr<XrdCl::AnyObject> response_mgr;
    response_mgr.reset(response);

    // Lock our state information then dispose of our object.
    auto l_state = m_state.lock();
    delete this;
    if (!l_state) {
      return;
    }

    // On function exit, notify any waiting threads.
    std::unique_ptr<char, std::function<void(char *)>> notify_guard(nullptr,
                                                                    [&](char *) { l_state->m_condvar.notify_all(); });

    {
      // On exit from the block, make sure m_status is set; it needs to be set before we notify threads.
      std::unique_ptr<char, std::function<void(char *)>> exit_guard(nullptr, [&](char *) {
        if (!l_state->m_status)
          l_state->m_status = std::make_unique<XrdCl::XRootDStatus>(XrdCl::stError, XrdCl::errInternal);
      });
      if (!status) {
        return;
      }
      if (status->IsOK()) {
        if (!response) {
          return;
        }
        XrdCl::Buffer *buf_ptr;
        response->Get(buf_ptr);
        // AnyObject::Set lacks specialization for nullptr
        response->Set(static_cast<int *>(nullptr));
        l_state->m_response.reset(buf_ptr);
      }
      l_state->m_status.reset(status);
    }
  }

  // Represents the current state of the callback.  The parent class only manages a weak_ptr
  // to the state.  If the asynchronous callback cannot lock the weak_ptr, then it assumes the
  // main thread has given up and doesn't touch any of the state variables.
  struct QueryAttrState {
    // Synchronize between the callback thread and the main thread; condvar predicate
    // is having m_status set.  m_mutex protects m_status.
    std::mutex m_mutex;
    std::condition_variable m_condvar;

    // Results from the server
    std::unique_ptr<XrdCl::XRootDStatus> m_status;
    std::unique_ptr<XrdCl::Buffer> m_response;
  };
  std::weak_ptr<QueryAttrState> m_state;
  XrdCl::FileSystem m_fs;
};

Source::Source(timespec now, std::unique_ptr<XrdCl::File> fh, const std::string &exclude)
    : m_lastDowngrade({0, 0}),
      m_id("(unknown)"),
      m_exclude(exclude),
      m_fh(std::move(fh)),
      m_stats(nullptr)
#ifdef XRD_FAKE_SLOW
      ,
      m_slow(++g_delayCount % XRD_SLOW_RATE == 0)
//, m_slow(++g_delayCount >= XRD_SLOW_RATE)
//, m_slow(true)
#endif
{
  if (m_fh.get()) {
    if (!m_fh->GetProperty("DataServer", m_id)) {
      edm::LogWarning("XrdFileWarning") << "Source::Source() failed to determine data server name.'";
    }
    if (m_exclude.empty()) {
      m_exclude = m_id;
    }
  }
  m_qm = QualityMetricFactory::get(now, m_id);
  m_prettyid = m_id + " (unknown site)";
  std::string domain_id;
  if (getDomain(m_id, domain_id)) {
    m_site = domain_id;
  } else {
    m_site = "Unknown (" + m_id + ")";
  }
  setXrootdSite();
  assert(m_qm.get());
  assert(m_fh.get());
  XrdSiteStatisticsInformation *statsService = XrdSiteStatisticsInformation::getInstance();
  if (statsService) {
    m_stats = statsService->getStatisticsForSite(m_site);
  }
}

bool Source::getHostname(const std::string &id, std::string &hostname) {
  size_t pos = id.find_last_of(':');
  hostname = id;
  if ((pos != std::string::npos) && (pos > 0)) {
    hostname = id.substr(0, pos);
  }

  bool retval = true;
  if (!hostname.empty() && ((hostname[0] == '[') || isdigit(hostname[0]))) {
    retval = false;
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    struct addrinfo *result;
    if (!getaddrinfo(hostname.c_str(), nullptr, &hints, &result)) {
      std::vector<char> host;
      host.reserve(256);
      if (!getnameinfo(result->ai_addr, result->ai_addrlen, &host[0], 255, nullptr, 0, NI_NAMEREQD)) {
        hostname = &host[0];
        retval = true;
      }
      freeaddrinfo(result);
    }
  }
  return retval;
}

bool Source::getDomain(const std::string &host, std::string &domain) {
  getHostname(host, domain);
  size_t pos = domain.find('.');
  if (pos != std::string::npos && (pos < domain.size())) {
    domain = domain.substr(pos + 1);
  }

  return !domain.empty();
}

bool Source::isDCachePool(XrdCl::File &file, const XrdCl::HostList *hostList) {
  // WORKAROUND: On open-file recovery in the Xrootd client, it'll carry around the
  // dCache opaque information to other sites, causing isDCachePool to erroneously return
  // true.  We are working with the upstream developers to solve this.
  //
  // For now, we see if the previous server also looks like a dCache pool - something that
  // wouldn't happen at a real site, as the previous server should look like a dCache door.
  std::string lastUrl;
  file.GetProperty("LastURL", lastUrl);
  if (!lastUrl.empty()) {
    bool result = isDCachePool(lastUrl);
    if (result && hostList && (hostList->size() > 1)) {
      if (isDCachePool((*hostList)[hostList->size() - 2].url.GetURL())) {
        return false;
      }
      return true;
    }
    return result;
  }
  return false;
}

bool Source::isDCachePool(const std::string &lastUrl) {
  XrdCl::URL url(lastUrl);
  XrdCl::URL::ParamsMap map = url.GetParams();
  // dCache pools always utilize this opaque identifier.
  if (map.find("org.dcache.uuid") != map.end()) {
    return true;
  }
  return false;
}

void Source::determineHostExcludeString(XrdCl::File &file, const XrdCl::HostList *hostList, std::string &exclude) {
  // Detect a dCache pool and, if we are in the federation context, give a custom
  // exclude parameter.
  // We assume this is a federation context if there's at least a regional, dCache door,
  // and dCache pool server (so, more than 2 servers!).

  exclude = "";
  if (hostList && (hostList->size() > 3) && isDCachePool(file, hostList)) {
    const XrdCl::HostInfo &info = (*hostList)[hostList->size() - 3];
    exclude = info.url.GetHostName();
    std::string lastUrl;
    file.GetProperty("LastURL", lastUrl);
    edm::LogVerbatim("XrdAdaptorInternal") << "Changing exclude list for URL " << lastUrl << " to " << exclude;
  }
}

bool Source::getXrootdSite(XrdCl::File &fh, std::string &site) {
  std::string lastUrl;
  fh.GetProperty("LastURL", lastUrl);
  if (lastUrl.empty() || isDCachePool(lastUrl)) {
    std::string server, id;
    if (!fh.GetProperty("DataServer", server)) {
      id = "(unknown)";
    } else {
      id = server;
    }
    if (lastUrl.empty()) {
      edm::LogWarning("XrdFileWarning") << "Unable to determine the URL associated with server " << id;
    }
    site = "Unknown";
    if (!server.empty()) {
      getDomain(server, site);
    }
    return false;
  }
  return getXrootdSiteFromURL(lastUrl, site);
}

bool Source::getXrootdSiteFromURL(std::string url, std::string &site) {
  const std::string attr = "sitename";
  XrdCl::Buffer *response = nullptr;
  XrdCl::Buffer arg(attr.size());
  arg.FromString(attr);

  std::string rsite;
  XrdCl::XRootDStatus st = QueryAttrHandler::query(url, "sitename", std::chrono::seconds(1), rsite);
  if (!st.IsOK()) {
    XrdCl::URL xurl(url);
    getDomain(xurl.GetHostName(), site);
    delete response;
    return false;
  }
  if (!rsite.empty() && (rsite[rsite.size() - 1] == '\n')) {
    rsite = rsite.substr(0, rsite.size() - 1);
  }
  if (rsite == "sitename") {
    XrdCl::URL xurl(url);
    getDomain(xurl.GetHostName(), site);
    return false;
  }
  site = rsite;
  return true;
}

void Source::setXrootdSite() {
  std::string site;
  bool goodSitename = getXrootdSite(*m_fh, site);
  if (!goodSitename) {
    edm::LogInfo("XrdAdaptorInternal") << "Xrootd server at " << m_id
                                       << " did not provide a sitename.  Monitoring may be incomplete.";
  } else {
    m_site = site;
    m_prettyid = m_id + " (site " + m_site + ")";
  }
  edm::LogInfo("XrdAdaptorInternal") << "Reading from new server " << m_id << " at site " << m_site;
}

Source::~Source() { new DelayedClose(fh(), m_id, m_site); }

std::shared_ptr<XrdCl::File> Source::getFileHandle() { return fh(); }

static void validateList(const XrdCl::ChunkList &cl) {
  off_t last_offset = -1;
  for (const auto &ci : cl) {
    assert(static_cast<off_t>(ci.offset) > last_offset);
    last_offset = ci.offset;
    assert(ci.length <= XRD_CL_MAX_CHUNK);
    assert(ci.offset < 0x1ffffffffff);
    assert(ci.offset > 0);
  }
  assert(cl.size() <= 1024);
}

void Source::handle(std::shared_ptr<ClientRequest> c) {
  edm::LogVerbatim("XrdAdaptorInternal") << "Reading from " << ID() << ", quality " << m_qm->get() << std::endl;
  c->m_source = shared_from_this();
  c->m_self_reference = c;
  m_qm->startWatch(c->m_qmw);
  if (m_stats) {
    std::shared_ptr<XrdReadStatistics> readStats = XrdSiteStatistics::startRead(stats(), c);
    c->setStatistics(readStats);
  }
#ifdef XRD_FAKE_SLOW
  if (m_slow)
    std::this_thread::sleep_for(std::chrono::milliseconds(XRD_DELAY));
#endif

  XrdCl::XRootDStatus status;
  if (c->m_into) {
    // See notes in ClientRequest definition to understand this voodoo.
    status = m_fh->Read(c->m_off, c->m_size, c->m_into, c.get());
  } else {
    XrdCl::ChunkList cl;
    cl.reserve(c->m_iolist->size());
    for (const auto &it : *c->m_iolist) {
      cl.emplace_back(it.offset(), it.size(), it.data());
    }
    validateList(cl);
    status = m_fh->VectorRead(cl, nullptr, c.get());
  }

  if (!status.IsOK()) {
    edm::Exception ex(edm::errors::FileReadError);
    ex << "XrdFile::Read or XrdFile::VectorRead failed with error: '" << status.ToStr() << "' (errNo = " << status.errNo
       << ")";
    ex.addContext("Calling Source::handle");
    throw ex;
  }
}
