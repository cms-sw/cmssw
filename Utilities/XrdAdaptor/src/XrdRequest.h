#ifndef Utilities_XrdAdaptor_XrdRequest_h
#define Utilities_XrdAdaptor_XrdRequest_h

#include <future>
#include <vector>

#include <XrdCl/XrdClXRootDResponses.hh>

#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include "QualityMetric.h"

namespace XrdAdaptor {

  class Source;

  class RequestManager;

  class XrdReadStatistics;

  class ClientRequest : public XrdCl::ResponseHandler {
    friend class Source;

  public:
    using IOPosBuffer = edm::storage::IOPosBuffer;
    using IOSize = edm::storage::IOSize;
    using IOOffset = edm::storage::IOOffset;

    ClientRequest(const ClientRequest &) = delete;
    ClientRequest &operator=(const ClientRequest &) = delete;

    ClientRequest(RequestManager &manager, void *into, IOSize size, IOOffset off)
        : m_failure_count(0), m_into(into), m_size(size), m_off(off), m_iolist(nullptr), m_manager(manager) {}

    ClientRequest(RequestManager &manager, std::shared_ptr<std::vector<IOPosBuffer>> iolist, IOSize size = 0)
        : m_failure_count(0), m_into(nullptr), m_size(size), m_off(0), m_iolist(iolist), m_manager(manager) {
      if (!m_iolist->empty() && !m_size) {
        for (edm::storage::IOPosBuffer const &buf : *m_iolist) {
          m_size += buf.size();
        }
      }
    }

    void setStatistics(std::shared_ptr<XrdReadStatistics> stats) { m_stats = stats; }

    ~ClientRequest() override;

    std::future<edm::storage::IOSize> get_future() { return m_promise.get_future(); }

    /**
     * Handle the response from the Xrootd server.
     */
    void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override;

    edm::storage::IOSize getSize() const { return m_size; }

    size_t getCount() const { return m_into ? 1 : m_iolist->size(); }

    /**
     * Returns a pointer to the current source; may be nullptr
     * if there is no outstanding IO
     */
    std::shared_ptr<Source const> getCurrentSource() const { return get_underlying_safe(m_source); }
    std::shared_ptr<Source> &getCurrentSource() { return get_underlying_safe(m_source); }

  private:
    std::shared_ptr<ClientRequest const> self_reference() const { return get_underlying_safe(m_self_reference); }
    std::shared_ptr<ClientRequest> &self_reference() { return get_underlying_safe(m_self_reference); }

    unsigned m_failure_count;
    void *m_into;
    edm::storage::IOSize m_size;
    edm::storage::IOOffset m_off;
    edm::propagate_const<std::shared_ptr<std::vector<edm::storage::IOPosBuffer>>> m_iolist;
    RequestManager &m_manager;
    edm::propagate_const<std::shared_ptr<Source>> m_source;
    edm::propagate_const<std::shared_ptr<XrdReadStatistics>> m_stats;

    // Some explanation is due here.  When an IO is outstanding,
    // Xrootd takes a raw pointer to this object.  Hence we cannot
    // allow it to go out of scope until some indeterminate time in the
    // future.  So, while the IO is outstanding, we take a reference to
    // ourself to prevent the object from being unexpectedly deleted.
    edm::propagate_const<std::shared_ptr<ClientRequest>> m_self_reference;

    std::promise<edm::storage::IOSize> m_promise;

    QualityMetricWatch m_qmw;
  };

}  // namespace XrdAdaptor

#endif
