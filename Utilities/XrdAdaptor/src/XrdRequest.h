#ifndef Utilities_XrdAdaptor_XrdRequest_h
#define Utilities_XrdAdaptor_XrdRequest_h

#include <future>
#include <vector>

#include <boost/utility.hpp>
#include <XrdCl/XrdClXRootDResponses.hh>

#include "Utilities/StorageFactory/interface/Storage.h"

#include "QualityMetric.h"

namespace XrdAdaptor {

class Source;

class RequestManager;

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

    ClientRequest(RequestManager &manager, std::shared_ptr<std::vector<IOPosBuffer> > iolist)
        : m_failure_count(0),
          m_into(nullptr),
          m_size(0),
          m_off(0),
          m_iolist(iolist),
          m_manager(manager)
    {
        // TODO: calculate size here.
    }

    virtual ~ClientRequest();

    std::future<IOSize> get_future()
    {
        return m_promise.get_future();
    }

    /**
     * Handle the response from the Xrootd server.
     */
    virtual void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override;

    IOSize getSize() const {return m_size;}

    /**
     * Returns a pointer to the current source; may be nullptr
     * if there is no outstanding IO
     */
    std::shared_ptr<Source> getCurrentSource() const {return m_source;}

private:
    unsigned m_failure_count;
    void *m_into;
    IOSize m_size;
    IOOffset m_off;
    std::shared_ptr<std::vector<IOPosBuffer> > m_iolist;
    RequestManager &m_manager;
    std::shared_ptr<Source> m_source;

    // Some explanation is due here.  When an IO is outstanding,
    // Xrootd takes a raw pointer to this object.  Hence we cannot
    // allow it to go out of scope until some indeterminate time in the
    // future.  So, while the IO is outstanding, we take a reference to
    // ourself to prevent the object from being unexpectedly deleted.
    std::shared_ptr<ClientRequest> m_self_reference;

    std::promise<IOSize> m_promise;

    QualityMetricWatch m_qmw;
};

}

#endif
