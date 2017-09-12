#ifndef Utilities_XrdAdaptor_XrdRequestManager_h
#define Utilities_XrdAdaptor_XrdRequestManager_h

#include <mutex>
#include <vector>
#include <set>
#include <condition_variable>
#include <random>
#include <sys/stat.h>

#include <boost/utility.hpp>
#include "tbb/concurrent_unordered_set.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include "XrdCl/XrdClFileSystem.hh"

#include "XrdRequest.h"
#include "XrdSource.h"

namespace XrdCl {
    class File;
}


namespace XrdAdaptor {

struct SourceHash {
    using Key =std::shared_ptr<Source>;
    size_t operator()(const Key& iKey) const {
      return tbb::tbb_hasher(iKey.get());
    }
  };

  
class XrootdException : public edm::Exception {

public:

    XrootdException(XrdCl::Status & xrootd_status, edm::Exception::Code code)
      : Exception(code), m_code(xrootd_status.code)
    {}

    ~XrootdException() noexcept override {};

    uint16_t getCode() { return m_code; }

private:

    uint16_t m_code;
};

class RequestManager : boost::noncopyable {

public:
    static const unsigned int XRD_DEFAULT_TIMEOUT = 3*60;

    virtual ~RequestManager() = default;

    /**
     * Interface for handling a client request.
     */
    std::future<IOSize> handle(void * into, IOSize size, IOOffset off)
    {
        auto c_ptr = std::make_shared<XrdAdaptor::ClientRequest>(*this, into, size, off);
        return handle(c_ptr);
    }

    std::future<IOSize> handle(std::shared_ptr<std::vector<IOPosBuffer> > iolist);

    /**
     * Handle a client request.
     * NOTE: The shared_ptr interface is required.  Depending on the state of the manager,
     * it may decide to issue multiple requests and return the first successful.  In that case,
     * some references to the client request may still be outstanding when this function returns.
     */
    std::future<IOSize> handle(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr);

    /**
     * Handle a failed client request.
     */
    void requestFailure(std::shared_ptr<XrdAdaptor::ClientRequest> c_ptr, XrdCl::Status &c_status);

    /**
     * Retrieve the names of the active sources
     * (primarily meant to enable meaningful log messages).
     */
    void getActiveSourceNames(std::vector<std::string> & sources) const;
    void getPrettyActiveSourceNames(std::vector<std::string> & sources) const;

    /**
     * Retrieve the names of the disabled sources
     * (primarily meant to enable meaningful log messages).
     */
    void getDisabledSourceNames(std::vector<std::string> & sources) const;

    /**
     * Return a pointer to an active file.  Useful for metadata
     * operations.
     */
    std::shared_ptr<XrdCl::File> getActiveFile() const;

    /**
     * Add the list of active connections to the exception extra info.
     */
    void addConnections(cms::Exception &) const;

    /**
     * Return current filename
     */
    const std::string & getFilename() const {return m_name;}

    /**
     * Some of the callback handlers need a weak_ptr reference to the RequestManager.
     * This allows the callback handler to know when it is OK to invoke RequestManager
     * methods.
     *
     * Hence, all instances need to be created through this factory function.
     */
    static std::shared_ptr<RequestManager>
    getInstance(const std::string &filename, XrdCl::OpenFlags::Flags flags, XrdCl::Access::Mode perms)
    {
        std::shared_ptr<RequestManager> instance(new RequestManager(filename, flags, perms));
        instance->initialize(instance);
        return instance;
    }

private:

    RequestManager(const std::string & filename, XrdCl::OpenFlags::Flags flags, XrdCl::Access::Mode perms);

    /**
     * Some of the callback handlers (particularly, file-open one) will want to call back into
     * the RequestManager.  However, the XrdFile may have already thrown away the reference.  Hence,
     * we need a weak_ptr to the original object before we can initialize.  This way, callback knows
     * to not reference the RequestManager.
     */
    void initialize(std::weak_ptr<RequestManager> selfref);

    /**
     * Handle the file-open response
     */
    virtual void handleOpen(XrdCl::XRootDStatus &status, std::shared_ptr<Source>);

    /**
     * Given a client request, split it into two requests lists.
     */
    void splitClientRequest(const std::vector<IOPosBuffer> &iolist,
                            std::vector<IOPosBuffer> &req1, std::vector<IOPosBuffer> &req2,
                            std::vector<std::shared_ptr<Source>> const& activeSources) const;

    /**
     * Given a request, broadcast it to all sources.
     * If active is true, broadcast is made to all active sources.
     * Otherwise, broadcast is made to the inactive sources.
     */
    void broadcastRequest(const ClientRequest &, bool active);

    /**
     * Check our set of active sources.
     * If necessary, this will kick off a search for a new source.
     * The source check is somewhat expensive so it is only done once every
     * second.
     */
    void checkSources(timespec &now, IOSize requestSize,
                      std::vector<std::shared_ptr<Source>>& activeSources,
                      std::vector<std::shared_ptr<Source>>& inactiveSources); // TODO: inline
    void checkSourcesImpl(timespec &now, IOSize requestSize,
                          std::vector<std::shared_ptr<Source>>& activeSources,
                          std::vector<std::shared_ptr<Source>>& inactiveSources);
    /**
     * Helper function for checkSources; compares the quality of source A
     * versus source B; if source A is significantly worse, remove it from
     * the list of active sources.
     *
     * NOTE: assumes two sources are active and the caller must already hold
     * m_source_mutex
     */
    bool compareSources(const timespec &now, unsigned a, unsigned b,
                        std::vector<std::shared_ptr<Source>>& activeSources,
                        std::vector<std::shared_ptr<Source>>& inactiveSources) const;

    /**
     * Anytime we potentially switch sources, update the internal site source list;
     * alert the user if necessary.
     */
    void reportSiteChange(std::vector<std::shared_ptr<Source> > const& iOld,
                        std::vector<std::shared_ptr<Source> > const& iNew,
                        std::string orig_site=std::string{}) const;

    /**
     * Update the StatisticsSenderService, if necessary, with the current server.
     */
    inline void updateCurrentServer();
    void queueUpdateCurrentServer(const std::string &);

    /**
     * Picks a single source for the next operation.
     */
    std::shared_ptr<Source> pickSingleSource();

    /**
     * Prepare an opaque string appropriate for asking a redirector to open the
     * current file but avoiding servers which we already have connections to.
     */
    std::string prepareOpaqueString() const;

    /**
     * Note these member variables can only be accessed when the source mutex
     * is held.
     */
    std::vector<std::shared_ptr<Source> > m_activeSources;
    std::vector<std::shared_ptr<Source> > m_inactiveSources;
  
    tbb::concurrent_unordered_set<std::string> m_disabledSourceStrings;
    tbb::concurrent_unordered_set<std::string> m_disabledExcludeStrings;
    tbb::concurrent_unordered_set<std::shared_ptr<Source>, SourceHash> m_disabledSources;

    // StatisticsSenderService wants to know what our current server is;
    // this holds last-successfully-opened server name
    std::atomic<std::string*> m_serverToAdvertise;

    timespec m_lastSourceCheck;
    int m_timeout;
    // If set to true, the next active source should be 1; 0 otherwise.
    bool m_nextInitialSourceToggle;
    // The time when the next active source check should be performed.
    timespec m_nextActiveSourceCheck;
    bool searchMode;

    const std::string m_name;
    XrdCl::OpenFlags::Flags m_flags;
    XrdCl::Access::Mode m_perms;
    mutable std::recursive_mutex m_source_mutex;

    std::mt19937 m_generator;
    std::uniform_real_distribution<float> m_distribution;

    std::atomic<unsigned> m_excluded_active_count;

    class OpenHandler : boost::noncopyable, public XrdCl::ResponseHandler {

    public:

        static std::shared_ptr<OpenHandler> getInstance(std::weak_ptr<RequestManager> manager)
        {
            OpenHandler *instance_ptr = new OpenHandler(manager);
            std::shared_ptr<OpenHandler> instance(instance_ptr);
            instance_ptr->m_self_weak = instance;
            return instance;
        }

        ~OpenHandler();

        /**
         * Handle the file-open response
         */
        virtual void HandleResponseWithHosts(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response, XrdCl::HostList *hostList) override;

        /**
         * Future-based version of the handler
         * If called while a file-open is in progress, we will not start a new file-open.
         * Instead, the callback will be fired for the ongoing open.
         *
         * NOTE NOTE: This function is not thread-safe due to a lock-ordering issue.
         * The caller must ensure it is not called from multiple threads at once
         * for this object.
         */
        std::shared_future<std::shared_ptr<Source> > open();

        /**
         * Returns the current source server name.  Useful primarily for debugging.
         */
        std::string current_source();

    private:

        OpenHandler(std::weak_ptr<RequestManager> manager);
        std::shared_future<std::shared_ptr<Source> > m_shared_future;
        std::promise<std::shared_ptr<Source> > m_promise;
        // Set to true only when there is an outstanding open request; not
        // protected by m_mutex, so the caller is required to know it is in a
        // thread-safe context.
        std::atomic<bool> m_outstanding_open {false};
        // Can only be touched when m_mutex is held.
        std::unique_ptr<XrdCl::File> m_file;
        std::recursive_mutex m_mutex;
        std::shared_ptr<OpenHandler> m_self;

        // Always maintain a weak self-reference; when the open is in-progress,
        // this is upgraded to a strong reference to prevent this object from
        // deletion as long as XrdCl has not performed the callback.
        std::weak_ptr<OpenHandler> m_self_weak;
        std::weak_ptr<RequestManager> m_manager;
    };

    std::shared_ptr<OpenHandler> m_open_handler;
};

}

#endif
