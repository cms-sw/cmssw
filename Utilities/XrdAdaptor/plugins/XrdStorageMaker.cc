
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/XrdAdaptor/src/XrdStatistics.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"

#include "XrdCl/XrdClDefaultEnv.hh"
#include "XrdNet/XrdNetUtils.hh"

#include <atomic>
#include <mutex>

namespace {

  class PrepareHandler : public XrdCl::ResponseHandler {
  public:
    PrepareHandler(const XrdCl::URL &url) : m_fs(url) { m_fileList.push_back(url.GetPath()); }

    void callAsyncPrepare() {
      auto status = m_fs.Prepare(m_fileList, XrdCl::PrepareFlags::Stage, 0, this);
      if (!status.IsOK()) {
        LogDebug("StageInError") << "XrdCl::FileSystem::Prepare submit failed with error '" << status.ToStr()
                                 << "' (errNo = " << status.errNo << ")";
        delete this;
      }
    }

    void HandleResponse(XrdCl::XRootDStatus *status, XrdCl::AnyObject *response) override {
      // Note: Prepare call has a response object.
      if (!status->IsOK()) {
        LogDebug("StageInError") << "XrdCl::FileSystem::Prepare failed with error '" << status->ToStr()
                                 << "' (errNo = " << status->errNo << ")";
      }
      delete response;
      delete status;
      delete this;
    }

  private:
    XrdCl::FileSystem m_fs;
    std::vector<std::string> m_fileList;
  };

}  // namespace

class XrdStorageMaker final : public StorageMaker {
public:
  static const unsigned int XRD_DEFAULT_TIMEOUT = 3*60;

  XrdStorageMaker():
  m_lastDebugLevel(1),//so that 0 will trigger change
  m_lastTimeout(0)
  {
    // When CMSSW loads, both XrdCl and XrdClient end up being loaded
    // (ROOT loads XrdClient).  XrdClient forces IPv4-only.  Accordingly,
    // we must explicitly set the default network stack in XrdCl to
    // whatever is available on the node (IPv4 or IPv6).
    XrdCl::Env *env = XrdCl::DefaultEnv::GetEnv();
    if (env)
    {
      env->PutString("NetworkStack", "IPAuto");
    }
    XrdNetUtils::SetAuto(XrdNetUtils::prefAuto);
    setTimeout(XRD_DEFAULT_TIMEOUT);
    setDebugLevel(0);
  }

  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  std::unique_ptr<Storage> open (const std::string &proto,
			 const std::string &path,
			 int mode,
       const AuxSettings& aux) const override
  {
    setDebugLevel(aux.debugLevel);
    setTimeout(aux.timeout);
    
    const StorageFactory *f = StorageFactory::get();
    StorageFactory::ReadHint readHint = f->readHint();
    StorageFactory::CacheHint cacheHint = f->cacheHint();

    if (readHint != StorageFactory::READ_HINT_UNBUFFERED
        || cacheHint == StorageFactory::CACHE_HINT_STORAGE)
      mode &= ~IOFlags::OpenUnbuffered;
    else
      mode |=  IOFlags::OpenUnbuffered;

    std::string fullpath(proto + ":" + path);
    auto file = std::make_unique<XrdFile>(fullpath, mode);
    return f->wrapNonLocalFile(std::move(file), proto, std::string(), mode);
  }

  void stagein (const std::string &proto, const std::string &path,
                        const AuxSettings& aux) const override
  {
    setDebugLevel(aux.debugLevel);
    setTimeout(aux.timeout);

    std::string fullpath(proto + ":" + path);
    XrdCl::URL url(fullpath);

    auto prep_handler = new PrepareHandler(url);
    prep_handler->callAsyncPrepare();
  }

  bool check (const std::string &proto,
		      const std::string &path,
          const AuxSettings& aux,
		      IOOffset *size = nullptr) const override
  {
    setDebugLevel(aux.debugLevel);
    setTimeout(aux.timeout);

    std::string fullpath(proto + ":" + path);
    XrdCl::URL url(fullpath);
    XrdCl::FileSystem fs(url);

    XrdCl::StatInfo *stat;
    if (!(fs.Stat(url.GetPath(), stat)).IsOK() || (stat == nullptr))
    {
        return false;
    }

    if (size) *size = stat->GetSize();
    return true;
  }

  void setDebugLevel (unsigned int level) const
  {
    auto oldLevel = m_lastDebugLevel.load();
    if(level == oldLevel) {
      return;
    }
    std::lock_guard<std::mutex> guard(m_envMutex);
    if(oldLevel != m_lastDebugLevel) {
      //another thread just changed this value
      return;
    }
    
    // 'Error' is way too low of debug level - we have interest
    // in warning in the default
    switch (level)
    {
      case 0:
        XrdCl::DefaultEnv::SetLogLevel("Warning");
        break;
      case 1:
        XrdCl::DefaultEnv::SetLogLevel("Info");
        break;
      case 2:
        XrdCl::DefaultEnv::SetLogLevel("Debug");
        break;
      case 3:
        XrdCl::DefaultEnv::SetLogLevel("Dump");
        break;
      case 4:
        XrdCl::DefaultEnv::SetLogLevel("Dump");
        break;
      default:
        edm::Exception ex(edm::errors::Configuration);
        ex << "Invalid log level specified " << level;
        ex.addContext("Calling XrdStorageMaker::setDebugLevel()");
        throw ex;
    }
    m_lastDebugLevel = level;
  }

  void setTimeout(unsigned int timeout) const
  {
    timeout = timeout ? timeout : XRD_DEFAULT_TIMEOUT;

    auto oldTimeout = m_lastTimeout.load();
    if (oldTimeout == timeout) {
      return;
    }
    
    std::lock_guard<std::mutex> guard(m_envMutex);
    if (oldTimeout != m_lastTimeout) {
      //Another thread beat us to changing the value
      return;
    }
    
    XrdCl::Env *env = XrdCl::DefaultEnv::GetEnv();
    if (env)
    {
      env->PutInt("StreamTimeout", timeout);
      env->PutInt("RequestTimeout", timeout);
      env->PutInt("ConnectionWindow", timeout);
      env->PutInt("StreamErrorWindow", timeout);
      // Crank down some of the connection defaults.  We have more
      // aggressive error recovery than the default client so we
      // can error out sooner.
      env->PutInt("ConnectionWindow", timeout/6+1);
      env->PutInt("ConnectionRetry", 2);
    }
    m_lastTimeout = timeout;
  }

private:
  mutable std::mutex m_envMutex;
  mutable std::atomic<unsigned int> m_lastDebugLevel;
  mutable std::atomic<unsigned int> m_lastTimeout;
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
DEFINE_FWK_SERVICE (XrdAdaptor::XrdStatisticsService);

