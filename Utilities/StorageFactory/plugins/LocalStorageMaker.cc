#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/File.h"
#include <unistd.h>
#include <sys/stat.h>

class LocalStorageMaker : public StorageMaker
{
public:
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode,
			 const std::string &tmpdir)
  {
    // Force unbuffered mode (bypassing page cache) off.  We
    // currently make so small reads that unbuffered access
    // will cause significant system load.  The unbuffered
    // hint is really for networked files (rfio, dcap, etc.),
    // where we don't want extra caching on client side due
    // non-sequential access patterns.
    mode &= ~IOFlags::OpenUnbuffered;
  
    return new File (path, mode); 
  }

  virtual bool check (const std::string &proto,
		      const std::string &path,
		      IOOffset *size = 0)
  {
    struct stat st;
    if (stat (path.c_str(), &st) != 0)
      return false;

    if (size)
      *size = st.st_size;

    return true;
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, LocalStorageMaker, "file");
