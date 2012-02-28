#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientUrlSet.hh"
#include "XrdClient/XrdClientEnv.hh"

class XrdStorageMaker : public StorageMaker
{
public:
  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode)
  {
    // The important part here is not the cache size (which will get
    // auto-adjusted), but the fact the cache is set to something non-zero.
    // If we don't do this before creating the XrdFile object, caching will be
    // completely disabled, resulting in poor performance.
    EnvPutInt(NAME_READCACHESIZE, 20*1024*1024);

    std::string fullpath(proto + ":" + path);
    return new XrdFile (fullpath, mode);
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
    EnvPutInt("DebugLevel", level);
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, XrdStorageMaker, "root");
