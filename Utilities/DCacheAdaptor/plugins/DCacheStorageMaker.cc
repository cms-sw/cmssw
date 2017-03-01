#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/DCacheAdaptor/interface/DCacheFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <unistd.h>
#include <dcap.h>

class DCacheStorageMaker : public StorageMaker
{
  /** Return appropriate path for use with dcap library.  If the path is in the
      URL form ([gsi]dcap://host:port/path), return the full URL as it is.  If
      the path is /pnfs form (dcap:/pnfs/path), return only the path part, unless
      the protocol is 'gsidcap', in which case return the whole thing.  */
  static std::string normalise (const std::string &proto, const std::string &path)
  {
    size_t p = path.find("/pnfs");
    if (p < 3)
      return (proto == "gsidcap") ? proto + ':' + path.substr(p) : path.substr(p);

    // remove multiple "/"
    p = path.find_first_not_of('/');

    // it's url, return the full thing
    return proto + "://" + path.substr(p);
  }

public:

  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual std::unique_ptr<Storage> open (const std::string &proto,
			 const std::string &path,
			 int mode,
       AuxSettings const& aux) const override
  {
    setTimeout(aux.timeout);
    const StorageFactory *f = StorageFactory::get();
    StorageFactory::ReadHint readHint = f->readHint();
    StorageFactory::CacheHint cacheHint = f->cacheHint();

    if (readHint != StorageFactory::READ_HINT_UNBUFFERED
	|| cacheHint == StorageFactory::CACHE_HINT_STORAGE)
      mode &= ~IOFlags::OpenUnbuffered;
    else
      mode |= IOFlags::OpenUnbuffered;

    auto file = std::make_unique<DCacheFile>(normalise(proto, path), mode);
    return f->wrapNonLocalFile(std::move(file), proto, std::string(), mode);
  }

  virtual void stagein (const std::string &proto,
		        const std::string &path,
            const AuxSettings& aux) const override
  {
    setTimeout(aux.timeout);
    std::string npath = normalise(proto, path);
    if (dc_stage(npath.c_str(), 0, 0) != 0) {
      cms::Exception ex("FileStageInError");
      ex << "Cannot stage in file '" << npath
	 << "', error was: " << dc_strerror(dc_errno)
	 << " (dc_errno=" << dc_errno << ")";
      ex.addContext("Calling DCacheStorageMaker::stagein()");
      throw ex;
    }
  }

  virtual bool check (const std::string &proto,
		      const std::string &path,
          const AuxSettings& aux,
		      IOOffset *size = 0) const override
  {
    setTimeout(aux.timeout);
    std::string testpath (normalise (proto, path));
    if (dc_access (testpath.c_str (), R_OK) != 0)
      return false;

    if (size)
    {
      struct stat64 buf;
      if (dc_stat64 (testpath.c_str (), &buf) != 0)
        return false;
	
      *size = buf.st_size;
    }

    return true;
  }
      
  private:

  void setTimeout(unsigned int timeout) const {
    if (timeout != 0) dc_setOpenTimeout(timeout);
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "dcache");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "dcap");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "gsidcap");
