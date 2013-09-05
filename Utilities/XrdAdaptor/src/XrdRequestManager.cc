
#include <assert.h>
#include <iostream>
#include <algorithm>

#include "XrdCl/XrdClFile.hh"

#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Utilities/XrdAdaptor/src/XrdRequestManager.h"

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

RequestManager::RequestManager(const std::string &filename, XrdCl::OpenFlags::Flags flags, XrdCl::Access::Mode perms)
    : m_nextInitialSourceToggle(false),
      m_name(filename),
      m_flags(flags),
      m_perms(perms),
      m_distribution(0,100),
      m_open_handler(*this)
{
  std::unique_ptr<XrdCl::File> file(new XrdCl::File());
  XrdCl::XRootDStatus status;
  if (! (status = file->Open(filename, flags, perms)).IsOK())
  {
    edm::Exception ex(edm::errors::FileOpenError);
    ex << "XrdCl::File::Open(name='" << filename
       << "', flags=0x" << std::hex << flags
       << ", permissions=0" << std::oct << perms << std::dec
       << ") => error '" << status.ToStr()
       << "' (errno=" << status.errNo << ", code=" << status.code << ")";
    ex.addContext("Calling XrdFile::open()");
    addConnections(ex);
    throw ex;
  }

  timespec ts;
  GET_CLOCK_MONOTONIC(ts);

  std::shared_ptr<Source> source(new Source(ts, std::move(file)));
  {
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    m_activeSources.push_back(source);
  }

  m_lastSourceCheck = ts;
  ts.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
  m_nextActiveSourceCheck = ts;
}

RequestManager::~RequestManager()
{}

void
RequestManager::checkSources(timespec &now, IOSize requestSize)
{
  edm::LogVerbatim("XrdAdaptorInternal") << "Time since last check "
    << timeDiffMS(now, m_lastSourceCheck) << "; last check "
    << m_lastSourceCheck.tv_sec << "; now " <<now.tv_sec
    << "; next check " << m_nextActiveSourceCheck.tv_sec << std::endl;  
  if (timeDiffMS(now, m_lastSourceCheck) > 1000 && timeDiffMS(now, m_nextActiveSourceCheck) > 0)
  {   
    checkSourcesImpl(now, requestSize);
  }
}

void
RequestManager::checkSourcesImpl(timespec &now, IOSize requestSize)
{
  std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);

  bool findNewSource = false;
  if (m_activeSources.size() <= 1)
    findNewSource = true;
  else if (m_activeSources.size() > 1)
  {
    edm::LogVerbatim("XrdAdaptorInternal") << "Source 0 quality " << m_activeSources[0]->getQuality() << ", source 1 quality " << m_activeSources[1]->getQuality() << std::endl;
    if ((m_activeSources[0]->getQuality() > 5130) ||
        ((m_activeSources[0]->getQuality() > 260) && (m_activeSources[1]->getQuality()*4 < m_activeSources[0]->getQuality())))
    {
        edm::LogWarning("XrdAdaptorInternal") << "Removing "
          << m_activeSources[0]->ID() << " from active sources due to poor quality ("
          << m_activeSources[0]->getQuality() << ")" << std::endl;
        if (m_activeSources[0]->getLastDowngrade().tv_sec != 0) findNewSource = true;
        m_activeSources[0]->setLastDowngrade(now);
        m_inactiveSources.emplace_back(m_activeSources[0]);
        m_activeSources.erase(m_activeSources.begin());
    }
    else if ((m_activeSources[1]->getQuality() > 5130) ||
        ((m_activeSources[1]->getQuality() > 260) && (m_activeSources[0]->getQuality()*4 < m_activeSources[1]->getQuality())))
    {
        edm::LogWarning("XrdAdaptorInternal") << "Removing "
          << m_activeSources[1]->ID() << " from active sources due to poor quality ("
          << m_activeSources[1]->getQuality() << ")" << std::endl;
        if (m_activeSources[1]->getLastDowngrade().tv_sec != 0) findNewSource = true;
        m_activeSources[1]->setLastDowngrade(now);
        m_inactiveSources.emplace_back(m_activeSources[1]);
        m_activeSources.erase(m_activeSources.begin()+1);
    }
    // NOTE: We could probably replace the copy with a better sort function at the cost of mental capacity.
    std::vector<std::shared_ptr<Source> > eligibleInactiveSources; eligibleInactiveSources.reserve(m_inactiveSources.size());
    for (const auto & source : m_inactiveSources) if (timeDiffMS(now, source->getLastDowngrade()) > (XRD_ADAPTOR_SHORT_OPEN_DELAY-1)*1000) eligibleInactiveSources.push_back(source);
    //for (const auto & source : m_inactiveSources) eligibleInactiveSources.push_back(source);
    std::vector<std::shared_ptr<Source> >::iterator bestInactiveSource = std::min_element(eligibleInactiveSources.begin(), eligibleInactiveSources.end(),
        [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
    std::vector<std::shared_ptr<Source> >::iterator worstActiveSource = std::max_element(m_activeSources.begin(), m_activeSources.end(),
        [](const std::shared_ptr<Source> &s1, const std::shared_ptr<Source> &s2) {return s1->getQuality() < s2->getQuality();});
    if (bestInactiveSource != eligibleInactiveSources.end() && bestInactiveSource->get())
    {
        edm::LogVerbatim("XrdAdaptorInternal") << "Best inactive source: " <<(*bestInactiveSource)->ID()
            << ", quality " << (*bestInactiveSource)->getQuality();
    }
    edm::LogVerbatim("XrdAdaptorInternal") << "Worst active source: " <<(*worstActiveSource)->ID() 
        << ", quality " << (*worstActiveSource)->getQuality();
    if ((bestInactiveSource != eligibleInactiveSources.end()) && m_activeSources.size() == 1)
    {
        m_activeSources.push_back(*bestInactiveSource);
        for (auto it = m_inactiveSources.begin(); it != m_inactiveSources.end(); it++) if (it->get() == bestInactiveSource->get()) {m_inactiveSources.erase(it); break;}
    }
    else while ((bestInactiveSource != eligibleInactiveSources.end()) && (*worstActiveSource)->getQuality() > (*bestInactiveSource)->getQuality()+XRD_ADAPTOR_SOURCE_QUALITY_FUDGE)
    {
        edm::LogVerbatim("XrdAdaptorInternal") << "Removing " << (*worstActiveSource)->ID()
            << " from active sources due to quality (" << (*worstActiveSource)->getQuality()
            << ") and promoting " << (*bestInactiveSource)->ID() << " (quality: "
            << (*bestInactiveSource)->getQuality() << ")" << std::endl;
        (*worstActiveSource)->setLastDowngrade(now);
        for (auto it = m_inactiveSources.begin(); it != m_inactiveSources.end(); it++) if (it->get() == bestInactiveSource->get()) {m_inactiveSources.erase(it); break;}
        m_inactiveSources.emplace_back(std::move(*worstActiveSource));
        m_activeSources.erase(worstActiveSource);
        m_activeSources.emplace_back(std::move(*bestInactiveSource));
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
    m_open_handler.open();
    m_lastSourceCheck = now;
  }

  now.tv_sec += XRD_ADAPTOR_SHORT_OPEN_DELAY;
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
  getActiveSourceNames(sources);
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

std::future<IOSize>
RequestManager::handle(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr)
{
  assert(c_ptr.get());
  timespec now;
  GET_CLOCK_MONOTONIC(now);
  checkSources(now, c_ptr->getSize());

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
  source->handle(c_ptr);
  return c_ptr->get_future();
}

std::string
RequestManager::prepareOpaqueString()
{
    std::lock_guard<std::recursive_mutex> sentry(m_source_mutex);
    std::stringstream ss;
    ss << "?tried=";
    size_t count = 0;
    for ( const auto & it : m_activeSources )
    {
        count++;
        ss << it->ID().substr(0, it->ID().find(":")) << ",";
    }
    for ( const auto & it : m_inactiveSources )
    {
        count++;
        ss << it->ID().substr(0, it->ID().find(":")) << ",";
    }
    for ( const auto & it : m_disabledSourceStrings )
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
        edm::LogVerbatim("XrdAdaptorInternal") << "Successfully opened new source: " << source->ID() << std::endl;

        if (m_activeSources.size() < 2)
        {
            m_activeSources.push_back(source);
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
                return b.get() + a.get();
            },
            std::move(future1),
            std::move(future2));
        timer.stop();
        //edm::LogVerbatim("XrdAdaptorInternal") << "Total time to create requests " << static_cast<int>(1000*timer.realTime()) << std::endl;
        return task;
    }
    if (req1->size()) return future1;
    if (req2->size()) return future2;

    std::promise<IOSize> p; p.set_value(0);
    return p.get_future();
}

void
RequestManager::requestFailure(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr)
{
    std::unique_lock<std::recursive_mutex> sentry(m_source_mutex);
    std::shared_ptr<Source> source_ptr = c_ptr->getCurrentSource();

    // Note that we do not delete the Source itself.  That is because this
    // function may be called from within XrdCl::ResponseHandler::HandleResponseWithHosts
    // In such a case, if you close a file in the handler, it will deadlock
    m_disabledSourceStrings.insert(source_ptr->ID());
    m_disabledSources.insert(source_ptr);

    if ((m_activeSources.size() > 0) && (m_activeSources[0].get() == source_ptr.get()))
    {
        m_activeSources.erase(m_activeSources.begin());
    }
    else if ((m_activeSources.size() > 1) && (m_activeSources[1].get() == source_ptr.get()))
    {
        m_activeSources.erase(m_activeSources.begin()+1);
    }
    std::shared_ptr<Source> new_source;
    if (m_activeSources.size() == 0)
    {
        std::shared_future<std::shared_ptr<Source> > future = m_open_handler.open();
        timespec now;
        GET_CLOCK_MONOTONIC(now);
        m_lastSourceCheck = now;
        // Note we only wait for 60 seconds here.  This is because we've already failed
        // once and the likelihood the program has some inconsistent state is decent.
        // We'd much rather fail hard than deadlock!
        sentry.unlock();
        std::future_status status = future.wait_for(std::chrono::seconds(60));
        if (status == std::future_status::timeout)
        {
            edm::Exception ex(edm::errors::FileOpenError);
            ex << "XrdCl::File::Open(name='" << m_name
               << "', flags=0x" << std::hex << m_flags
               << ", permissions=0" << std::oct << m_perms << std::dec
               << ", old source=" << source_ptr->ID()
               << ") => timeout when waiting for file open";
            ex.addContext("In XrdAdaptor::RequestManager::requestFailure()");
            addConnections(ex);
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
                throw;
            }
        }
        sentry.lock();
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
        if (io.size() > chunksize)
        {
            IOSize newsize = io.size() - chunksize;
            IOOffset newoffset = io.offset() + chunksize;
            void* newdata = static_cast<char*>(io.data()) + chunksize;
            output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
            io.set_offset(newoffset);
            io.set_data(newdata);
            io.set_size(newsize);
            chunksize = 0;
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
        if (io.size() > chunksize)
        {
            IOSize newsize = io.size() - chunksize;
            IOOffset newoffset = io.offset() + chunksize;
            void* newdata = static_cast<char*>(io.data()) + chunksize;
            output.emplace_back(IOPosBuffer(io.offset(), io.data(), chunksize));
            io.set_offset(newoffset);
            io.set_data(newdata);
            io.set_size(newsize);
            chunksize = 0;
        }
        else
        {
            output.push_back(io);
            chunksize -= io.size();
            input.pop_back();
        }
    }
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
    chunk1 = static_cast<float>(XRD_CL_MAX_CHUNK)*(q2/(q1+q2));
    chunk2 = static_cast<float>(XRD_CL_MAX_CHUNK)*(q1/(q1+q2));

    while (tmp_iolist.size()-front > 0)
    {
        consumeChunkFront(front, tmp_iolist, req1, chunk1);
        consumeChunkBack(front, tmp_iolist, req2, chunk2);
    }

    IOSize size1 = 0, size2 = 0, size_orig = 0;
    for (const auto & it : iolist) size_orig += it.size();
    for (const auto & it : req1) size1 += it.size();
    for (const auto & it : req2) size2 += it.size();

    assert(size_orig == size1 + size2);

    edm::LogVerbatim("XrdAdaptorInternal") << "Original request size " << iolist.size() << " (" << size_orig << " bytes) split into requests size " << req1.size() << " (" << size1 << " bytes) and " << req2.size() << " (" << size2 << " bytes)" << std::endl;
}

XrdAdaptor::RequestManager::OpenHandler::OpenHandler(RequestManager & manager)
  : m_manager(manager)
{
}

void
XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response, XrdCl::HostList *hostList)
{
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);
    if (status->IsOK())
    {
        timespec now;
        GET_CLOCK_MONOTONIC(now);
        std::shared_ptr<Source> source(new Source(now, std::move(m_file)));
        m_promise.set_value(source);
        m_manager.handleOpen(*status, source);
    }
    else
    {
        m_file.reset();
        std::shared_ptr<Source> emptySource;
        edm::Exception ex(edm::errors::FileOpenError);
        ex << "XrdCl::File::Open(name='" << m_manager.m_name
           << "', flags=0x" << std::hex << m_manager.m_flags
           << ", permissions=0" << std::oct << m_manager.m_perms << std::dec
           << ") => error '" << status->ToStr()
           << "' (errno=" << status->errNo << ", code=" << status->code << ")";
        ex.addContext("In XrdAdaptor::RequestManager::OpenHandler::HandleResponseWithHosts()");
        m_manager.addConnections(ex);

        m_promise.set_exception(std::make_exception_ptr(ex));
        m_manager.handleOpen(*status, emptySource);
    }
    delete status;
    delete hostList;
}

std::shared_future<std::shared_ptr<Source> >
XrdAdaptor::RequestManager::OpenHandler::open()
{
    std::lock_guard<std::recursive_mutex> sentry(m_mutex);

    if (m_file.get())
    {
        return m_shared_future;
    }
    std::promise<std::shared_ptr<Source> > new_promise;
    m_promise.swap(new_promise);
    m_shared_future = m_promise.get_future().share();

    auto opaque = m_manager.prepareOpaqueString();
    std::string new_name = m_manager.m_name + opaque;
    edm::LogVerbatim("XrdAdaptorInternal") << "Trying to open URL: " << new_name;
    m_file.reset(new XrdCl::File());
    XrdCl::XRootDStatus status;
    if (!(status = m_file->Open(new_name, m_manager.m_flags, m_manager.m_perms, this)).IsOK())
    {
      edm::Exception ex(edm::errors::FileOpenError);
      ex << "XrdCl::File::Open(name='" << new_name
         << "', flags=0x" << std::hex << m_manager.m_flags
         << ", permissions=0" << std::oct << m_manager.m_perms << std::dec
         << ") => error '" << status.ToStr()
         << "' (errno=" << status.errNo << ", code=" << status.code << ")";
      ex.addContext("Calling XrdAdaptor::RequestManager::OpenHandler::open()");
      m_manager.addConnections(ex);
      throw ex;
    }
    return m_shared_future;
}

