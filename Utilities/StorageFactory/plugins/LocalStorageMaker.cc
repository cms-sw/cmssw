#define _GNU_SOURCE 1
#define _FILE_OFFSET_BITS 64
#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/File.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

class LocalStorageMaker : public StorageMaker
{
public:
  virtual std::unique_ptr<Storage> open (const std::string &proto,
			 const std::string &path,
			 int mode,
       const AuxSettings&) const override
    {
      const StorageFactory *f = StorageFactory::get();
      StorageFactory::ReadHint readHint = f->readHint();
      StorageFactory::CacheHint cacheHint = f->cacheHint();

      if (readHint != StorageFactory::READ_HINT_UNBUFFERED
	  || cacheHint == StorageFactory::CACHE_HINT_STORAGE)
	mode &= ~IOFlags::OpenUnbuffered;
      else
	mode |= IOFlags::OpenUnbuffered;

      auto file = std::make_unique<File> (path, mode);
      return f->wrapNonLocalFile (std::move(file), proto, path, mode);
    }

  virtual bool check (const std::string &/*proto*/,
		      const std::string &path,
          const AuxSettings&,
		      IOOffset *size = 0) const override
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
