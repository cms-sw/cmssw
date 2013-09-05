
#include "FWCore/Utilities/interface/EDMException.h"

#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"

// These are to be removed once the new client supports prepare requests.
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientUrlSet.hh"
#include "XrdCl/XrdClDefaultEnv.hh"


class XrdStorageMaker : public StorageMaker
{
public:
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
    switch (level)
    {
      case 0:
        XrdCl::DefaultEnv::SetLogLevel("Error");
        break;
      case 1:
        XrdCl::DefaultEnv::SetLogLevel("Warning");
        break;
      case 2:
        XrdCl::DefaultEnv::SetLogLevel("Info");
        break;
      case 3:
        XrdCl::DefaultEnv::SetLogLevel("Debug");
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
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
