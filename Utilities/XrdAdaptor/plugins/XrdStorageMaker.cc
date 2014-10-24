#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientUrlSet.hh"
#include "XrdClient/XrdClientEnv.hh"

class XrdStorageMaker : public StorageMaker
{
private:
  unsigned int timeout_ = 0;

public:
  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode) override
  {
    // The important part here is not the cache size (which will get
    // auto-adjusted), but the fact the cache is set to something non-zero.
    // If we don't do this before creating the XrdFile object, caching will be
    // completely disabled, resulting in poor performance.
    EnvPutInt(NAME_READCACHESIZE, 20*1024*1024);

    // XrdClient has various timeouts which vary from 3 minutes to 8 hours.
    // This enforces an even default (10 minutes) more appropriate for the
    // cmsRun case.
    if (timeout_ <= 0) {setTimeout(600);}

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
    XrdClientAdmin admin(fullpath.c_str());
    if (admin.Connect())
    {
      XrdOucString str(fullpath.c_str());
      XrdClientUrlSet url(str);
      admin.Prepare(url.GetFile().c_str(), kXR_stage | kXR_noerrs, 0);
    }
  }

  virtual bool check (const std::string &proto,
		      const std::string &path,
		      IOOffset *size = 0) override
  {
    std::string fullpath(proto + ":" + path);
    XrdClientAdmin admin(fullpath.c_str());
    if (! admin.Connect())
      return false; // FIXME: Throw?

    long      id;
    long      flags;
    long      modtime;
    long long xrdsize;

    XrdOucString str(fullpath.c_str());
    XrdClientUrlSet url(str);

    if (! admin.Stat(url.GetFile().c_str(), id, xrdsize, flags, modtime))
      return false; // FIXME: Throw?

    *size = xrdsize;
    return true;
  }

  virtual void setDebugLevel (unsigned int level) override
  {
    EnvPutInt("DebugLevel", level);
  }

  virtual void setTimeout(unsigned int timeout) override
  {
    timeout_ = timeout;
    if (timeout == 0) {return;}
    EnvPutInt("ConnectTimeout", timeout/3+1); // Default 120.  This should allow multiple connections to timeout before the open fails.
    EnvPutInt("RequestTimeout", timeout/3+1); // Default 300.  This should allow almost three requests to be performed before the transaction times out.
    EnvPutInt("TransactionTimeout", timeout); // Default 28800

    // Safety mechanism - if client is redirected more than 255 times in 600
    // seconds, then we abort the interaction.
    EnvPutInt("RedirCntTimeout", 600); // Default 36000

    // Enforce some CMS defaults.
    EnvPutInt("MaxRedirectcount", 32); // Default 16
    EnvPutInt("ReconnectWait", 5); // Default 5
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
