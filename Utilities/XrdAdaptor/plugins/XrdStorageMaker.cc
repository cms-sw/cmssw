#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"

// These are to be removed once the new client supports prepare requests.
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientUrlSet.hh"
#include "XrdCl/XrdClDefaultEnv.hh"
// We muck with internal symbols of XrdCl to avoid duplicate definition issues
// See https://github.com/xrootd/xrootd/issues/16
#define __XRD_CL_OPTIMIZERS_HH__
#include "XrdCl/XrdClLog.hh"


class XrdStorageMaker : public StorageMaker
{
public:
  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode)
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

  virtual void stagein (const std::string &proto, const std::string &path)
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
		      IOOffset *size = 0)
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

  virtual void setDebugLevel (unsigned int level)
  {
    XrdCl::Log *log = XrdCl::DefaultEnv::GetLog();
    log->SetLevel(static_cast<XrdCl::Log::LogLevel>(level+2));
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
