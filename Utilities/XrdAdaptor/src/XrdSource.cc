
// See http://stackoverflow.com/questions/12523122/what-is-glibcxx-use-nanosleep-all-about
#define _GLIBCXX_USE_NANOSLEEP
#include <thread>
#include <chrono>
#include <iostream>
#include <assert.h>

#include "XrdCl/XrdClFile.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "XrdSource.h"
#include "XrdRequest.h"
#include "QualityMetric.h"

#define MAX_REQUEST 256*1024

#ifdef XRD_FAKE_SLOW
//#define XRD_DELAY 5140
#define XRD_DELAY 1000
#define XRD_SLOW_RATE 2
int g_delayCount = 0;
#else
int g_delayCount = 0;
#endif

using namespace XrdAdaptor;

Source::Source(timespec now, std::unique_ptr<XrdCl::File> fh)
    : m_lastDowngrade({0, 0}),
      m_id(fh.get() ? fh->GetDataServer() : "(unknown)"),
      m_fh(std::move(fh)),
      m_qm(QualityMetricFactory::get(now, m_id))
#ifdef XRD_FAKE_SLOW
    , m_slow(++g_delayCount % XRD_SLOW_RATE == 0)
    //, m_slow(++g_delayCount >= XRD_SLOW_RATE)
    //, m_slow(true)
#endif
{
    assert(m_qm.get());
    assert(m_fh.get());
    m_buffer.reserve(MAX_REQUEST);
}

Source::~Source()
{
  XrdCl::XRootDStatus status;
  if (! (status = m_fh->Close()).IsOK())
    edm::LogWarning("XrdFileWarning")
      << "Source::~Source() failed with error '" << status.ToString()
      << "' (errno=" << status.errNo << ", code=" << status.code << ")";
  m_fh.reset();
}

std::shared_ptr<XrdCl::File>
Source::getFileHandle()
{
    return m_fh;
}

void
Source::handle(std::shared_ptr<ClientRequest> c)
{
    edm::LogVerbatim("XrdAdaptorInternal") << "Reading from " << ID() << ", quality " << m_qm->get() << std::endl;
    c->m_source = shared_from_this();
    c->m_self_reference = c;
    m_qm->startWatch(c->m_qmw);
#ifdef XRD_FAKE_SLOW
    if (m_slow) std::this_thread::sleep_for(std::chrono::milliseconds(XRD_DELAY));
#endif
    if (c->m_into)
    {
        // See notes in ClientRequest definition to understand this voodoo.
        m_fh->Read(c->m_off, c->m_size, c->m_into, c.get());
    }
    else
    {
        XrdCl::ChunkList cl;
        cl.reserve(c->m_iolist->size());
        for (const auto & it : *c->m_iolist)
        {
            XrdCl::ChunkInfo ci(it.offset(), it.size(), it.data());
            cl.emplace_back(ci);
        }
        m_fh->VectorRead(cl, nullptr, c.get());
    }
}

