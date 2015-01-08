
#include "FWCore/Utilities/interface/EDMException.h"

#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"

#include "XrdCl/XrdClDefaultEnv.hh"
#include "XrdNet/XrdNetUtils.hh"

class MakerResponseHandler : public XrdCl::ResponseHandler
{
public:
    virtual void HandleResponse( XrdCl::XRootDStatus *status,
                                 XrdCl::AnyObject    *response )
    {
        if (response) delete response;
    }

};

class XrdStorageMaker : public StorageMaker
{
public:
  static const unsigned int XRD_DEFAULT_TIMEOUT = 3*60;

  XrdStorageMaker()
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
  }

  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode) override
  {

    StorageFactory *f = StorageFactory::get();
    StorageFactory::ReadHint readHint = f->readHint();
    StorageFactory::CacheHint cacheHint = f->cacheHint();

    if (readHint != StorageFactory::READ_HINT_UNBUFFERED
        || cacheHint == StorageFactory::CACHE_HINT_STORAGE)
      mode &= ~IOFlags::OpenUnbuffered;
    else
      mode |=  IOFlags::OpenUnbuffered;

    std::string fullpath(proto + ":" + path);
    Storage *file = new XrdFile (fullpath, mode);
    return f->wrapNonLocalFile(file, proto, std::string(), mode);
  }

  virtual void stagein (const std::string &proto, const std::string &path) override
  {
    std::string fullpath(proto + ":" + path);
    XrdCl::URL url(fullpath);
    XrdCl::FileSystem fs(url);
    std::vector<std::string> fileList; fileList.push_back(url.GetPath());
    fs.Prepare(fileList, XrdCl::PrepareFlags::Stage, 0, &m_null_handler);
  }

  virtual bool check (const std::string &proto,
		      const std::string &path,
		      IOOffset *size = 0) override
  {
    std::string fullpath(proto + ":" + path);
    XrdCl::URL url(fullpath);
    XrdCl::FileSystem fs(url);

    XrdCl::StatInfo *stat;
    if (!(fs.Stat(fullpath, stat)).IsOK() || (stat == nullptr))
    {
        return false;
    }

    if (size) *size = stat->GetSize();
    return true;
  }

  virtual void setDebugLevel (unsigned int level) override
  {
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
  }

  virtual void setTimeout(unsigned int timeout) override
  {
    timeout = timeout ? timeout : XRD_DEFAULT_TIMEOUT;
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
  }

private:
  MakerResponseHandler m_null_handler;
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
