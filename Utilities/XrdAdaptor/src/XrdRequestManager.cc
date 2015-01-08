
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <netdb.h>

#include "XrdCl/XrdClFile.hh"
#include "XrdCl/XrdClDefaultEnv.hh"

#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"

#include "Utilities/XrdAdaptor/src/XrdRequestManager.h"
#include "Utilities/XrdAdaptor/src/XrdHostHandler.hh"

#define XRD_CL_MAX_CHUNK 512*1024

#define XRD_ADAPTOR_SHORT_OPEN_DELAY 5

#ifdef XRD_FAKE_OPEN_PROBE
#define XRD_ADAPTOR_OPEN_PROBE_PERCENT 100
#define XRD_ADAPTOR_LONG_OPEN_DELAY 20
// This is the minimal difference in quality required to swap an active and inactive source
#define XRD_ADAPTOR_SOURCE_QUALITY_FUDGE 0
#else
#define XRD_ADAPTOR_OPEN_PROBE_PERCENT 10
#define XRD_ADAPTOR_LONG_OPEN_DELAY 2*60
#define XRD_ADAPTOR_SOURCE_QUALITY_FUDGE 100
#endif

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#define GET_CLOCK_MONOTONIC(ts) \
{ \
  clock_serv_t cclock; \
  mach_timespec_t mts; \
  host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock); \
  clock_get_time(cclock, &mts); \
  mach_port_deallocate(mach_task_self(), cclock); \
  ts.tv_sec = mts.tv_sec; \
  ts.tv_nsec = mts.tv_nsec; \
}
#else
#define GET_CLOCK_MONOTONIC(ts) \
  clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

using namespace XrdAdaptor;

long long timeDiffMS(const timespec &a, const timespec &b)
{
  long long diff = (a.tv_sec - b.tv_sec) * 1000;
  diff += (a.tv_nsec - b.tv_nsec) / 1e6;
  return diff;
}

/*
 * We do not care about the response of sending the monitoring information;
 * this handler class simply frees any returned buffer to prevent memory leaks.
 */
class SendMonitoringInfoHandler : boost::noncopyable, public XrdCl::ResponseHandler
{
    virtual void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override
    {
        if (response)
        {
            XrdCl::Buffer *buffer = nullptr;
            response->Get(buffer);
            delete buffer;
        }
    }
};

SendMonitoringInfoHandler nullHandler;


static void
SendMonitoringInfo(XrdCl::File &file)
{
        // Do not send this to a dCache data server as they return an error.
        // In some versions of dCache, sending the monitoring information causes
        // the server to close the connection - resulting in failures.
    if (Source::isDCachePool(file)) {return;}

    // Send the monitoring info, if available.
    const char * jobId = edm::storage::StatisticsSenderService::getJobID();
    std::string lastUrl;
    file.GetProperty("LastURL", lastUrl);
    if (jobId && lastUrl.size())
    {
        XrdCl::FileSystem fs = XrdCl::FileSystem(XrdCl::URL(lastUrl));
        fs.SendInfo(jobId, &nullHandler, 30);
        edm::LogInfo("XrdAdaptorInternal") << "Set monitoring ID to " << jobId << ".";
    }
}


RequestManager::RequestManager(const std::string &filename, XrdCl::OpenFlags::Flags flags, XrdCl::Access::Mode perms)
    : m_timeout(XRD_DEFAULT_TIMEOUT),
      m_nextInitialSourceToggle(false),
      m_name(filename),
      m_flags(flags),
      m_perms(perms),
      m_distribution(0,100),
      m_excluded_active_count(0)
{
}


void
RequestManager::initialize(std::weak_ptr<RequestManager> self)
{
  m_open_handler = OpenHandler::getInstance(self);

  XrdCl::Env *env = XrdCl::DefaultEnv::GetEnv();
  if (env) {env->GetInt("StreamErrorWindow", m_timeout);}

  std::string orig_site;
  if (!Source::getXrootdSiteFromURL(m_name, orig_site) && (orig_site.find(".") == std::string::npos))
  {
    std::string hostname;
    if (Source::getHostname(orig_site, hostname))
    {
      Source::getDomain(hostname, orig_site);
    }
  }

  std::unique_ptr<XrdCl::File> file;
  edm::Exception ex(edm::errors::FileOpenError);
  bool validFile = false;
  const int retries = 5;
  std::string excludeString;
  for (int idx=0; idx<retries; idx++)
  {
    file.reset(new XrdCl::File());
    auto opaque = prepareOpaqueString();
    std::string new_filename = m_name + (opaque.size() ? ((m_name.find("?") == m_name.npos) ? "?" : "&") + opaque : "");
    SyncHostResponseHandler handler;
    XrdCl::XRootDStatus openStatus = file->Open(new_filename, m_flags, m_perms, &handler);
    if (!openStatus.IsOK())
    { // In this case, we failed immediately - this indicates we have previously tried to talk to this
      // server and it was marked bad - xrootd couldn't even queue up the request internally!
      // In practice, we obsere this happening when the call to getXrootdSiteFromURL fails due to the
      // redirector being down or authentication failures.
      ex.clearMessage();
      ex.clearContext();
      ex.clearAdditionalInfo();
      ex << "XrdCl::File::Open(name='" << m_name
         << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec
         << ") => error '" << openStatus.ToStr()
         << "' (errno=" << openStatus.errNo << ", code=" << openStatus.code << ")";
      ex.addContext("Calling XrdFile::open()");
      ex.addAdditionalInfo("Remote server already encountered a fatal error; no redirections were performed.");
      throw ex;
    }
    handler.WaitForResponse();
    std::unique_ptr<XrdCl::XRootDStatus> status = handler.GetStatus();
    std::unique_ptr<XrdCl::HostList> hostList = handler.GetHosts();
    Source::determineHostExcludeString(*file, hostList.get(), excludeString);
    assert(status);
    if (status->IsOK())
    {
      validFile = true;
      break;
    }
    else
    {
      ex.clearMessage();
      ex.clearContext();
      ex.clearAdditionalInfo();
      ex << "XrdCl::File::Open(name='" << m_name
         << "', flags=0x" << std::hex << m_flags
         << ", permissions=0" << std::oct << m_perms << std::dec
         << ") => error '" << status->ToStr()
         << "' (errno=" << status->errNo << ", code=" << status->code << ")";
      ex.addContext("Calling XrdFile::open()");
      addConnections(ex);
      std::string dataServer, lastUrl;
      file->GetProperty("DataServer", dataServer);
      file->GetProperty("LastURL", lastUrl);
      if (dataServer.size())
      {
        ex.addAdditionalInfo("Problematic data server: " + dataServer);
      }
      if (lastUrl.size())
      {
        ex.addAdditionalInfo("Last URL tried: " + lastUrl);
        edm::LogWarning("XrdAdaptorInternal") << "Failed to open file at URL " << lastUrl << ".";
      }
      if (std::find(m_disabledSourceStrings.begin(), m_disabledSourceStrings.end(), dataServer) != m_disabledSourceStrings.end())
      {
        ex << ". No additional data servers were found.";
        throw ex;
      }
      if (dataServer.size())
      {
        m_disabledSourceStrings.insert(dataServer);
        m_disabledExcludeStrings.insert(excludeString);
      }
      // In this case, we didn't go anywhere - we stayed at the redirector and it gave us a file-not-found.
      if (lastUrl == new_filename)
      {
        edm::LogWarning("XrdAdaptorInternal") << lastUrl << ", " << new_filename;
        throw ex;
      }
    }
  }
  if (!validFile)
  {
      throw ex;
  }
  SendMonitoringInfo(*file);

  timespec ts;
  GET_CLOCK_MONOTONIC(ts);

  std::shared_ptr<Source> source(new Source(ts, std::move(file), excludeString));
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    m_activeSources.push_back(source);
    updateSiteInfo(orig_site);
  }

  m_lastSourceCheck = ts;
  ts.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
  m_nextActiveSourceCheck = ts;
}


void
RequestManager::updateSiteInfo(std::string orig_site)
{
  std::string siteA, siteB, siteList;
  if (m_activeSources.size()) {siteA = m_activeSources[0]->Site();}
  if (m_activeSources.size() == 2) {siteB = m_activeSources[1]->Site();}
  siteList = siteA;
  if (siteB.size() && (siteB != siteA)) {siteList = siteA + ", " + siteB;}
  if (orig_site.size() && (orig_site != siteList))
  {
    edm::LogWarning("XrdAdaptor") << "Data is served from " << siteList << " instead of original site " << orig_site;
    m_activeSites = siteList;
  }
  else if (!orig_site.size() && (siteList != m_activeSites))
  {
    edm::LogWarning("XrdAdaptor") << "Data is now served from " << siteList << " instead of previous " << m_activeSites;
    m_activeSites = siteList;
  }
}


void
RequestManager::checkSources(timespec &now, IOSize requestSize)
{
  edm::LogVerbatim("XrdAdaptorInternal") << "Time since last check "
    << timeDiffMS(now, m_lastSourceCheck) << "; last check "
    << m_lastSourceCheck.tv_sec << "; now " <<now.tv_sec
    << "; next check " << m_nextActiveSourceCheck.tv_sec << std::endl;  
  if (timeDiffMS(now, m_lastSourceCheck) > 1000)
  {
    { // Be more aggressive about getting rid of very bad sources.
      std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);   
      compareSources(now, 0, 1);
      compareSources(now, 1, 0);
    }
    if (timeDiffMS(now, m_nextActiveSourceCheck) > 0)
    {
      checkSourcesImpl(now, requestSize);
    }
  }
}


bool
RequestManager::compareSources(const timespec &now, unsigned a, unsigned b)
{
  if (m_activeSources.size() < std::max(a, b)+1) {return false;}

  bool findNewSource = false;
  if ((m_activeSources[a]->getQuality() > 5130) ||
     ((m_activeSources[a]->getQuality() > 260) && (m_activeSources[b]->getQuality()*4 < m_activeSources[a]->getQuality())))
  {
    edm::LogVerbatim("XrdAdaptorInternal") << "Removing "
          << m_activeSources[a]->PrettyID() << " from active sources due to poor quality ("
          << m_activeSources[a]->getQuality() << " vs " << m_activeSources[b]->getQuality() << ")" << std::endl;
    if (m_activeSources[a]->getLastDowngrade().tv_sec != 0) {findNewSource = true;}
    m_activeSources[a]->setLastDowngrade(now);
    m_inactiveSources.emplace_back(m_activeSources[a]);
    m_activeSources.erase(m_activeSources.begin()+a);
    updateSiteInfo();
  }
  return findNewSource;
}

void
RequestManager::checkSourcesImpl(timespec &now, IOSize requestSize)
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);

  bool findNewSource = false;
  if (m_activeSources.size() <= 1)
  {
    findNewSource = true;
  }
  else if (m_activeSources.size() > 1)
  {
    edm::LogVerbatim("XrdAdaptorInternal") << "Source 0 quality " << m_activeSources[0]->getQuality() << ", source 1 quality " << m_activeSources[1]->getQuality() << std::endl;
    findNewSource |= compareSources(now, 0, 1);
    findNewSource |= compareSources(now, 1, 0);

    // NOTE: We could probably replace the copy with a better sort function.
    // However, there are typically very few sources and the correctness is more obvious right now.
    std::vector<std::shared_ptr<Source> > eligibleInactiveSources; eligibleInactiveSources.reserve(m_inactiveSources.size());
    for (const auto & source : m_inactiveSources)
    {
      if (timeDiffMS(now, source->getLastDowngrade()) > (XRD_ADAPTOR_SHORT_OPEN_DELAY-1)*1000) {eligibleInactiveSources.push_back(source);}
    }
    std::vector<std::shared_ptr<Source> >::iterator bestInactiveSource = std::min_element(eligibleInactiveSources.begin(), eligibleInactiveSources.end(),
        [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
    std::vector<std::shared_ptr<Source> >::iterator worstActiveSource = std::max_element(m_activeSources.begin(), m_activeSources.end(),
        [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
    if (bestInactiveSource != eligibleInactiveSources.end() && bestInactiveSource->get())
    {
      edm::LogVerbatim("XrdAdaptorInternal") << "Best inactive source: " <<(*bestInactiveSource)->PrettyID()
            << ", quality " << (*bestInactiveSource)->getQuality();
    }
    edm::LogVerbatim("XrdAdaptorInternal") << "Worst active source: " <<(*worstActiveSource)->PrettyID() 
        << ", quality " << (*worstActiveSource)->getQuality();
        // Only upgrade the source if we only have one source and the best inactive one isn't too horrible.
        // Regardless, we will want to re-evaluate the new source quickly (within 5s).
    if ((bestInactiveSource != eligibleInactiveSources.end()) && m_activeSources.size() == 1 && ((*bestInactiveSource)->getQuality() < 4*m_activeSources[0]->getQuality()))
    {
        m_activeSources.push_back(*bestInactiveSource);
        updateSiteInfo();
        for (auto it = m_inactiveSources.begin(); it != m_inactiveSources.end(); it++) if (it->get() == bestInactiveSource->get()) {m_inactiveSources.erase(it); break;}
    }
    else while ((bestInactiveSource != eligibleInactiveSources.end()) && (*worstActiveSource)->getQuality() > (*bestInactiveSource)->getQuality()+XRD_ADAPTOR_SOURCE_QUALITY_FUDGE)
    {
        edm::LogVerbatim("XrdAdaptorInternal") << "Removing " << (*worstActiveSource)->PrettyID()
            << " from active sources due to quality (" << (*worstActiveSource)->getQuality()
            << ") and promoting " << (*bestInactiveSource)->PrettyID() << " (quality: "
            << (*bestInactiveSource)->getQuality() << ")" << std::endl;
        (*worstActiveSource)->setLastDowngrade(now);
        for (auto it = m_inactiveSources.begin(); it != m_inactiveSources.end(); it++) if (it->get() == bestInactiveSource->get()) {m_inactiveSources.erase(it); break;}
        m_inactiveSources.emplace_back(std::move(*worstActiveSource));
        m_activeSources.erase(worstActiveSource);
        m_activeSources.emplace_back(std::move(*bestInactiveSource));
        updateSiteInfo();
        eligibleInactiveSources.clear();
        for (const auto & source : m_inactiveSources) if (timeDiffMS(now, source->getLastDowngrade()) > (XRD_ADAPTOR_LONG_OPEN_DELAY-1)*1000) eligibleInactiveSources.push_back(source);
        bestInactiveSource = std::min_element(eligibleInactiveSources.begin(), eligibleInactiveSources.end(),
            [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
        worstActiveSource = std::max_element(m_activeSources.begin(), m_activeSources.end(),
            [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
    }
    if (!findNewSource && (timeDiffMS(now, m_lastSourceCheck) > 1000*XRD_ADAPTOR_LONG_OPEN_DELAY))
    {
        float r = m_distribution(m_generator);
        if (r < XRD_ADAPTOR_OPEN_PROBE_PERCENT)
        {
            findNewSource = true;
        }
    }
  }
  if (findNewSource)
  {
    m_open_handler->open();
    m_lastSourceCheck = now;
  }

  // Only aggressively look for new sources if we don't have two.
  if (m_activeSources.size() == 2)
  {
    now.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
  }
  else
  {
    now.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
  }
  m_nextActiveSourceCheck = now;
}

std::shared_ptr<XrdCl::File>
RequestManager::getActiveFile()
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  return m_activeSources[0]->getFileHandle();
}

void
RequestManager::getActiveSourceNames(std::vector<std::string> & sources)
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  sources.reserve(m_activeSources.size());
  for (auto const& source : m_activeSources) {
    sources.push_back(source->ID());
  }
}

void
RequestManager::getPrettyActiveSourceNames(std::vector<std::string> & sources)
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  sources.reserve(m_activeSources.size());
  for (auto const& source : m_activeSources) {
    sources.push_back(source->PrettyID());
  }
}

void
RequestManager::getDisabledSourceNames(std::vector<std::string> & sources)
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
  sources.reserve(m_disabledSourceStrings.size());
  for (auto const& source : m_disabledSourceStrings) {
    sources.push_back(source);
  }
}

void
RequestManager::addConnections(cms::Exception &ex)
{
  std::vector<std::string> sources;
  getPrettyActiveSourceNames(sources);
  for (auto const& source : sources)
  {
    ex.addAdditionalInfo("Active source: " + source);
  }
  sources.clear();
  getDisabledSourceNames(sources);
  for (auto const& source : sources)
  {
    ex.addAdditionalInfo("Disabled source: " + source);
  }
}

std::shared_ptr<Source>
RequestManager::pickSingleSource()
{
  std::shared_ptr<Source> source = nullptr;
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    if (m_activeSources.size() == 2)
    {
        if (m_nextInitialSourceToggle)
        {
            source = m_activeSources[0];
            m_nextInitialSourceToggle = false;
        }
        else
        {
            source = m_activeSources[1];
            m_nextInitialSourceToggle = true;
        }
    }
    else
    {
        source = m_activeSources[0];
    }
  }
  return source;
}

std::future<IOSize>
RequestManager::handle(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr)
{
  assert(c_ptr.get());
  timespec now;
  GET_CLOCK_MONOTONIC(now);
  checkSources(now, c_ptr->getSize());

  std::shared_ptr<Source> source = pickSingleSource();
  source->handle(c_ptr);
  return c_ptr->get_future();
}

std::string
RequestManager::prepareOpaqueString()
{
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    std::stringstream ss;
    ss << "tried=";
    size_t count = 0;
    for ( const auto & it : m_activeSources )
    {
        count++;
        ss << it->ExcludeID().substr(0, it->ExcludeID().find(":")) << ",";
    }
    for ( const auto & it : m_inactiveSources )
    {
        count++;
        ss << it->ExcludeID().substr(0, it->ExcludeID().find(":")) << ",";
    }
    for ( const auto & it : m_disabledExcludeStrings )
    {
        count++;
        ss << it.substr(0, it.find(":")) << ",";
    }
    if (count)
    {
        std::string tmp_str = ss.str();
        return tmp_str.substr(0, tmp_str.size()-1);
    }
    return "";
}

void 
XrdAdaptor::RequestManager::handleOpen(XrdCl::XRootDStatus &status, std::shared_ptr<Source> source)
{
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    if (status.IsOK())
    {
        edm::LogVerbatim("XrdAdaptorInternal") << "Successfully opened new source: " << source->PrettyID() << std::endl;
        for (const auto & s : m_activeSources)
        {
            if (source->ID() == s->ID())
            {
                edm::LogVerbatim("XrdAdaptorInternal") << "Xrootd server returned excluded source " << source->PrettyID()
                    << "; ignoring" << std::endl;
                unsigned returned_count = ++m_excluded_active_count;
                m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
                if (returned_count >= 3) {m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - 2*XRD_ADAPTOR_SHORT_OPEN_DELAY;}
                return;
            }
        }
        for (const auto & s : m_inactiveSources)
        {
            if (source->ID() == s->ID())
            {
                edm::LogVerbatim("XrdAdaptorInternal") << "Xrootd server returned excluded inactive source " << source->PrettyID() 
                    << "; ignoring" << std::endl;
                m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
                return;
            }
        }
        if (m_activeSources.size() < 2)
        {
            m_activeSources.push_back(source);
            updateSiteInfo();
        }
        else
        {
            m_inactiveSources.push_back(source);
        }
    }
    else
    {   // File-open failure - wait at least 120s before next attempt.
        edm::LogVerbatim("XrdAdaptorInternal") << "Got failure when trying to open a new source" << std::endl;
        m_nextActiveSourceCheck.tv_sec += XRD_ADAPTOR_LONG_OPEN_DELAY - XRD_ADAPTOR_SHORT_OPEN_DELAY;
    }
}

std::future<IOSize>
XrdAdaptor::RequestManager::handle(std::shared_ptr<std::vector<IOPosBuffer> > iolist)
{
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);

    timespec now;
    GET_CLOCK_MONOTONIC(now);

    edm::CPUTimer timer;
    timer.start();

    assert(m_activeSources.size());
    if (m_activeSources.size() == 1)
    {
        std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr(new XrdAdaptor::ClientRequest(*this, iolist));
        checkSources(now, c_ptr->getSize());
        m_activeSources[0]->handle(c_ptr);
        return c_ptr->get_future();
    }

    assert(iolist.get());
    std::shared_ptr<std::vector<IOPosBuffer> > req1(new std::vector<IOPosBuffer>);
    std::shared_ptr<std::vector<IOPosBuffer> > req2(new std::vector<IOPosBuffer>);
    splitClientRequest(*iolist, *req1, *req2);

    checkSources(now, req1->size() + req2->size());
    // CheckSources may have removed a source
    if (m_activeSources.size() == 1)
    {
        std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr(new XrdAdaptor::ClientRequest(*this, iolist));
        m_activeSources[0]->handle(c_ptr);
        return c_ptr->get_future();
    }

    std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr1, c_ptr2;
    std::future<IOSize> future1, future2;
    if (req1->size())
    {
        c_ptr1.reset(new XrdAdaptor::ClientRequest(*this, req1));
        m_activeSources[0]->handle(c_ptr1);
        future1 = c_ptr1->get_future();
    }
    if (req2->size())
    {
        c_ptr2.reset(new XrdAdaptor::ClientRequest(*this, req2));
        m_activeSources[1]->handle(c_ptr2);
        future2 = c_ptr2->get_future();
    }
    if (req1->size() && req2->size())
    {
        std::future<IOSize> task = std::async(std::launch::deferred,
            [](std::future<IOSize> a, std::future<IOSize> b){
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
                b.wait(); a.wait();
                return b.get() + a.get();
            },
            std::move(future1),
            std::move(future2));
        timer.stop();
        //edm::LogVerbatim("XrdAdaptorInternal") << "Total time to create requests " << static_cast<int>(1000*timer.realTime()) << std::endl;
        return task;
    }
    else if (req1->size()) { return future1; }
    else if (req2->size()) { return future2; }
    else
    {   // Degenerate case - no bytes to read.
        std::promise<IOSize> p; p.set_value(0);
        return p.get_future();
    }
}

void
RequestManager::requestFailure(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr, XrdCl::Status &c_status)
{
    std::unique_lock<std::recursive_mutex> sentry(m_source_mutex);
    std::shared_ptr<Source> source_ptr = c_ptr->getCurrentSource();

    // Fail early for invalid responses - XrdFile has a separate path for handling this.
    if (c_status.code == XrdCl::errInvalidResponse)
    {
        edm::LogWarning("XrdAdaptorInternal") << "Invalid response when reading from " << source_ptr->PrettyID();
        XrootdException ex(c_status, edm::errors::FileReadError);
        ex << "XrdAdaptor::RequestManager::requestFailure readv(name='" << m_name
               << "', flags=0x" << std::hex << m_flags
               << ", permissions=0" << std::oct << m_perms << std::dec
               << ", old source=" << source_ptr->PrettyID()
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

    if ((m_activeSources.size() > 0) && (m_activeSources[0].get() == source_ptr.get()))
    {
        m_activeSources.erase(m_activeSources.begin());
        updateSiteInfo();
    }
    else if ((m_activeSources.size() > 1) && (m_activeSources[1].get() == source_ptr.get()))
    {
        m_activeSources.erase(m_activeSources.begin()+1);
        updateSiteInfo();
    }
    std::shared_ptr<Source> new_source;
    if (m_activeSources.size() == 0)
    {
        std::shared_future<std::shared_ptr<Source> > future = m_open_handler->open();
        timespec now;
        GET_CLOCK_MONOTONIC(now);
        m_lastSourceCheck = now;
        // Note we only wait for 180 seconds here.  This is because we've already failed
        // once and the likelihood the program has some inconsistent state is decent.
        // We'd much rather fail hard than deadlock!
        sentry.unlock();
        std::future_status status = future.wait_for(std::chrono::seconds(m_timeout+10));
        if (status == std::future_status::timeout)
        {
            XrootdException ex(c_status, edm::errors::FileOpenError);
            ex << "XrdAdaptor::RequestManager::requestFailure Open(name='" << m_name
               << "', flags=0x" << std::hex << m_flags
               << ", permissions=0" << std::oct << m_perms << std::dec
               << ", old source=" << source_ptr->PrettyID()
               << ", current server=" << m_open_handler->current_source()
               << ") => timeout when waiting for file open";
            ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
            addConnections(ex);
            throw ex;
        }
        else
        {
            try
            {
                new_source = future.get();
            }
            catch (edm::Exception &ex)
            {
                ex.addContext("Handling XrdAdaptor::RequestManager::requestFailure()");
                ex.addAdditionalInfo("Original failed source is " + source_ptr->PrettyID());
                throw;
            }
        }
        sentry.lock();
        
        if (std::find(m_disabledSourceStrings.begin(), m_disabledSourceStrings.end(), new_source->ID()) != m_disabledSourceStrings.end())
        {
            // The server gave us back a data node we requested excluded.  Fatal!
            XrootdException ex(c_status, edm::errors::FileOpenError);
            ex << "XrdAdaptor::RequestManager::requestFailure Open(name='" << m_name
               << "', flags=0x" << std::hex << m_flags
               << ", permissions=0" << std::oct << m_perms << std::dec
               << ", old source=" << source_ptr->PrettyID()
               << ", new source=" << new_source->PrettyID() << ") => Xrootd server returned an excluded source";
            ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
            addConnections(ex);
            throw ex;
        }
        m_activeSources.push_back(new_source);
        updateSiteInfo();
    }
    else
    {
        new_source = m_activeSources[0];
    }
    new_source->handle(c_ptr);
}

static void
consumeChunkFront(size_t &front, std::vector<IOPosBuffer> &input, std::vector<IOPosBuffer> &output, IOSize chunksize)
{
    while ((chunksize > 0) && (front < input.size()))
    {
        IOPosBuffer &io = input[front];
        IOPosBuffer &outio = output.back();
        if (io.size() > chunksize)
        {
            IOSize consumed;
            if (output.size() && (outio.size() < XRD_CL_MAX_CHUNK) && (outio.offset() + static_cast<IOOffset>(outio.size()) == io.offset()))
            {
                if (outio.size() + chunksize > XRD_CL_MAX_CHUNK)
                {
                    consumed = (XRD_CL_MAX_CHUNK - outio.size());
                    outio.set_size(XRD_CL_MAX_CHUNK);
                }
                else
                {
                    consumed = chunksize;
                    outio.set_size(outio.size() + consumed);
                }
            }
            else
            {
                consumed = chunksize;
                output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
            }
            chunksize -= consumed;
            IOSize newsize = io.size() - consumed;
            IOOffset newoffset = io.offset() + consumed;
            void* newdata = static_cast<char*>(io.data()) + consumed;
            io.set_offset(newoffset);
            io.set_data(newdata);
            io.set_size(newsize);
        }
        else if (io.size() == 0)
        {
            front++;
        }
        else
        {
            output.push_back(io);
            chunksize -= io.size();
            front++;
        }
    }
}

static void
consumeChunkBack(size_t front, std::vector<IOPosBuffer> &input, std::vector<IOPosBuffer> &output, IOSize chunksize)
{
    while ((chunksize > 0) && (front < input.size()))
    {
        IOPosBuffer &io = input.back();
        IOPosBuffer &outio = output.back();
        if (io.size() > chunksize)
        {
            IOSize consumed;
            if (output.size() && (outio.size() < XRD_CL_MAX_CHUNK) && (outio.offset() + static_cast<IOOffset>(outio.size()) == io.offset()))
            {
                if (outio.size() + chunksize > XRD_CL_MAX_CHUNK)
                {
                    consumed = (XRD_CL_MAX_CHUNK - outio.size());
                    outio.set_size(XRD_CL_MAX_CHUNK);
                }
                else
                {
                    consumed = chunksize;
                    outio.set_size(outio.size() + consumed);
                }
            }
            else
            {
                consumed = chunksize;
                output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
            }
            chunksize -= consumed;
            IOSize newsize = io.size() - consumed;
            IOOffset newoffset = io.offset() + consumed;
            void* newdata = static_cast<char*>(io.data()) + consumed;
            io.set_offset(newoffset);
            io.set_data(newdata);
            io.set_size(newsize);
        }
        else if (io.size() == 0)
        {
            input.pop_back();
        }
        else
        {
            output.push_back(io);
            chunksize -= io.size();
            input.pop_back();
        }
    }
}

static IOSize validateList(const std::vector<IOPosBuffer> req)
{
    IOSize total = 0;
    off_t last_offset = -1;
    for (const auto & it : req)
    {
        total += it.size();
        assert(it.offset() > last_offset);
        last_offset = it.offset();
        assert(it.size() <= XRD_CL_MAX_CHUNK);
        assert(it.offset() < 0x1ffffffffff);
    }
    return total;
}

void
XrdAdaptor::RequestManager::splitClientRequest(const std::vector<IOPosBuffer> &iolist, std::vector<IOPosBuffer> &req1, std::vector<IOPosBuffer> &req2)
{
    if (iolist.size() == 0) return;
    std::vector<IOPosBuffer> tmp_iolist(iolist.begin(), iolist.end());
    req1.reserve(iolist.size()/2+1);
    req2.reserve(iolist.size()/2+1);
    size_t front=0;

    float q1 = static_cast<float>(m_activeSources[0]->getQuality());
    float q2 = static_cast<float>(m_activeSources[1]->getQuality());
    IOSize chunk1, chunk2;
    chunk1 = static_cast<float>(XRD_CL_MAX_CHUNK)*(q2*q2/(q1*q1+q2*q2));
    chunk2 = static_cast<float>(XRD_CL_MAX_CHUNK)*(q1*q1/(q1*q1+q2*q2));

    while (tmp_iolist.size()-front > 0)
    {
        consumeChunkFront(front, tmp_iolist, req1, chunk1);
        consumeChunkBack(front, tmp_iolist, req2, chunk2);
    }
    std::sort(req1.begin(), req1.end(), [](const IOPosBuffer & left, const IOPosBuffer & right){return left.offset() < right.offset();});
    std::sort(req2.begin(), req2.end(), [](const IOPosBuffer & left, const IOPosBuffer & right){return left.offset() < right.offset();});

    IOSize size1 = validateList(req1);
    IOSize size2 = validateList(req2);
    IOSize size_orig = 0;
    for (const auto & it : iolist) size_orig += it.size();

    assert(size_orig == size1 + size2);

    edm::LogVerbatim("XrdAdaptorInternal") << "Original request size " << iolist.size() << " (" << size_orig << " bytes) split into requests size " << req1.size() << " (" << size1 << " bytes) and " << req2.size() << " (" << size2 << " bytes)" << std::endl;
}

XrdAdaptor::RequestManager::OpenHandler::OpenHandler(std::weak_ptr<RequestManager> manager)
  : m_manager(manager)
{
}


    // Cannot use ~OpenHandler=default as XrdCl::File is not fully
    // defined in the header.
XrdAdaptor::RequestManager::OpenHandler::~OpenHandler()
{
}


void
XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts(XrdCl::XRootDStatus *status_ptr, XrdCl::AnyObject *, XrdCl::HostList *hostList_ptr)
{
  std::shared_ptr<Source> source;
  std::unique_ptr<XrdCl::XRootDStatus> status(status_ptr);
  std::unique_ptr<XrdCl::HostList> hostList(hostList_ptr);

    // Make sure we get rid of the strong self-reference when the callback finishes.
  std::shared_ptr<OpenHandler> self = m_self;
  m_self.reset();

  auto manager = m_manager.lock();
    // Manager object has already been deleted.  Cleanup the
    // response objects, remove our self-reference, and ignore the response.
  if (!manager)
  {
    return;
  }
  {
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);

    if (status->IsOK())
    {
        SendMonitoringInfo(*m_file);
        timespec now;
        GET_CLOCK_MONOTONIC(now);

        std::string excludeString;
        Source::determineHostExcludeString(*m_file, hostList.get(), excludeString);

        source.reset(new Source(now, std::move(m_file), excludeString));
        m_promise.set_value(source);
    }
    else
    {
        m_file.reset();
        edm::Exception ex(edm::errors::FileOpenError);
        ex << "XrdCl::File::Open(name='" << manager->m_name
           << "', flags=0x" << std::hex << manager->m_flags
           << ", permissions=0" << std::oct << manager->m_perms << std::dec
           << ") => error '" << status->ToStr()
           << "' (errno=" << status->errNo << ", code=" << status->code << ")";
        ex.addContext("In XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts()");
        manager->addConnections(ex);

        m_promise.set_exception(std::make_exception_ptr(ex));
    }
  }
  manager->handleOpen(*status, source);
}

std::string
XrdAdaptor::RequestManager::OpenHandler::current_source()
{
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);

    if (!m_file.get())
    {
        return "(no open in progress)";
    }
    std::string dataServer;
    m_file->GetProperty("DataServer", dataServer);
    if (!dataServer.size()) { return "(unknown source)"; }
    return dataServer;
}

std::shared_future<std::shared_ptr<Source> >
XrdAdaptor::RequestManager::OpenHandler::open()
{
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);
    auto manager_ptr = m_manager.lock();
    if (!manager_ptr)
    {
      edm::Exception ex(edm::errors::LogicError);
      ex << "XrdCl::File::Open() =>"
         << " error: OpenHandler called within an invalid RequestManager context."
         << "  This is a logic error and should be reported to the CMSSW developers.";
      ex.addContext("Calling XrdAdaptor::RequestManager::OpenHandler::open()");
      throw ex;
    }
    RequestManager &manager = *manager_ptr;
    auto self_ptr = m_self_weak.lock();
    if (!self_ptr)
    {
      edm::Exception ex(edm::errors::LogicError);
      ex << "XrdCl::File::Open() => error: "
         << "OpenHandler called after it was deleted.  This is a logic error "
         << "and should be reported to the CMSSW developers.";
      ex.addContext("Calling XrdAdapter::RequestManager::OpenHandler::open()");
      throw ex;
    }

    if (m_file.get())
    {
        return m_shared_future;
    }
    std::promise<std::shared_ptr<Source> > new_promise;
    m_promise.swap(new_promise);
    m_shared_future = m_promise.get_future().share();

    auto opaque = manager.prepareOpaqueString();
    std::string new_name = manager.m_name + ((manager.m_name.find("?") == manager.m_name.npos) ? "?" : "&") + opaque;
    edm::LogVerbatim("XrdAdaptorInternal") << "Trying to open URL: " << new_name;
    m_file.reset(new XrdCl::File());
    XrdCl::XRootDStatus status;
    if (!(status = m_file->Open(new_name, manager.m_flags, manager.m_perms, this)).IsOK())
    {
      edm::Exception ex(edm::errors::FileOpenError);
      ex << "XrdCl::File::Open(name='" << new_name
         << "', flags=0x" << std::hex << manager.m_flags
         << ", permissions=0" << std::oct << manager.m_perms << std::dec
         << ") => error '" << status.ToStr()
         << "' (errno=" << status.errNo << ", code=" << status.code << ")";
      ex.addContext("Calling XrdAdaptor::RequestManager::OpenHandler::open()");
      manager.addConnections(ex);
      throw ex;
    }
      // Have a strong self-reference for as long as the callback is in-progress.
    m_self = self_ptr;
    return m_shared_future;
}

