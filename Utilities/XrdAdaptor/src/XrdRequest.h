#ifndef Utilities_XrdAdaptor_XrdRequest_h
#define Utilities_XrdAdaptor_XrdRequest_h

#include <future>
#include <vector>

#include <boost/utility.hpp>
#include <XrdCl/XrdClXRootDResponses.hh>

#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include "QualityMetric.h"

namespace XrdAdaptor {

class Source;

class RequestManager;

class XrdReadStatistics;

class ClientRequest : boost::noncopyable, public XrdCl::ResponseHandler {

friend class Source;

public:

    ClientRequest(RequestManager &manager, void *into, IOSize size, IOOffset off)
        : m_failure_count(0),
          m_into(into),
          m_size(size),
          m_off(off),
          m_iolist(nullptr),
          m_manager(manager)
    {
    }

    ClientRequest(RequestManager &manager, std::shared_ptr<std::vector<IOPosBuffer> > iolist, IOSize size=0)
        : m_failure_count(0),
          m_into(nullptr),
          m_size(size),
          m_off(0),
          m_iolist(iolist),
          m_manager(manager)
    {
        if (!m_iolist->empty() && !m_size)
        {
            for (IOPosBuffer const & buf : *m_iolist)
            {
                m_size += buf.size();
            }
        }
    }

    void setStatistics(std::shared_ptr<XrdReadStatistics> stats)
    {
        m_stats = stats;
    }

    ~ClientRequest() override;

    std::future<IOSize> get_future()
    {
        return m_promise.get_future();
    }

    /**
     * Handle the response from the Xrootd server.
     */
    void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override;

    IOSize getSize() const {return m_size;}

    size_t getCount() const {return m_into ? 1 : m_iolist->size();}

    /**
     * Returns a pointer to the current source; may be nullptr
     * if there is no outstanding IO
     */
    std::shared_ptr<Source const> getCurrentSource() const {return get_underlying_safe(m_source);}
    std::shared_ptr<Source>& getCurrentSource() {return get_underlying_safe(m_source);}

private:
    std::shared_ptr<ClientRequest const> self_reference() const {return get_underlying_safe(m_self_reference);}
    std::shared_ptr<ClientRequest>& self_reference() {return get_underlying_safe(m_self_reference);}

    unsigned m_failure_count;
    void *m_into;
    IOSize m_size;
    IOOffset m_off;
    edm::propagate_const<std::shared_ptr<std::vector<IOPosBuffer>>> m_iolist;
    RequestManager &m_manager;
    edm::propagate_const<std::shared_ptr<Source>> m_source;
    edm::propagate_const<std::shared_ptr<XrdReadStatistics>> m_stats;

    // Some explanation is due here.  When an IO is outstanding,
    // Xrootd takes a raw pointer to this object.  Hence we cannot
    // allow it to go out of scope until some indeterminate time in the
    // future.  So, while the IO is outstanding, we take a reference to
    // ourself to prevent the object from being unexpectedly deleted.
    edm::propagate_const<std::shared_ptr<ClientRequest>> m_self_reference;

    std::promise<IOSize> m_promise;

    QualityMetricWatch m_qmw;
};

}

#endif
