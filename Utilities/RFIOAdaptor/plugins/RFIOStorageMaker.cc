#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"
#include "Utilities/StorageFactory/interface/LocalCacheFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIO.h"
#include "Utilities/RFIOAdaptor/interface/RFIOPluginFactory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstdlib>
#include "shift/stager_api.h"

class RFIOStorageMaker : public StorageMaker
{
  /** Normalise new RFIO TURL style.  Handle most obvious mis-spellings
      like excess '/' characters and /dpm vs. /castor syntax
      differences.  */
  std::string normalise (const std::string &path)
  {
    std::string prefix;
    // look for options
    size_t suffix = path.find("?");
    if (suffix == std::string::npos)
    {
      // convert old syntax to new but leave "host:/path" alone
      suffix = 0;
      if (path.find (":") == std::string::npos)
      {
        size_t e = path.find_first_not_of ("/");
        if (e != std::string::npos)
        {
          size_t c = path.find("/castor/");
          if ((c != std::string::npos) && (c == e-1))
	  {
	    // /castor/path -> rfio:///?path=/castor/path
	    suffix = c;
	    prefix = "rfio:///?path=";
          }
          else
	  {
            c = path.find("/dpm/");
            if ((c != std::string::npos) && (c == e-1))
	    {
	      // /dpm/path -> rfio:///dpm/path
	      suffix = c;
	      prefix = "rfio://";
            }
	  }
        }
      }
    }
    else
    {
      // new syntax, leave alone except normalize host
      prefix = path.substr(0, suffix);
      size_t h = prefix.find_first_not_of('/');
      size_t s = prefix.find_last_not_of('/');
      prefix.resize(s+1);
      prefix.replace(0,h,"rfio://");
      prefix += '/';
    }

    return prefix + path.substr(suffix);
  }

public:
  RFIOStorageMaker()
  {
    std::string rfiotype("");
    bool err = false;
    try
    {
      edm::Service<edm::SiteLocalConfig> siteconfig;
      if (!siteconfig.isAvailable())
        err = true;
      else
        rfiotype = siteconfig->rfioType();
    }
    catch (const cms::Exception &e)
    {
      err = true;
    }

    if (err)
      edm::LogWarning("RFIOStorageMaker") 
        << "SiteLocalConfig Failed: SiteLocalConfigService is not loaded yet."
        << "Will use default 'castor' RFIO implementation.";

    if (rfiotype.empty())
      rfiotype = "castor";

    // Force Castor to move control messages out of client TCP buffers faster.
    putenv("RFIO_TCP_NODELAY=1");

    RFIOPluginFactory::get()->create(rfiotype);
    Cthread_init();
  }

  virtual Storage *open (const std::string &proto,
		         const std::string &path,
			 int mode,
			 const std::string &tmpdir)
  {
    StorageFactory *f = StorageFactory::get();
    StorageFactory::ReadHint readHint = f->readHint();
    StorageFactory::CacheHint cacheHint = f->cacheHint();

    if (readHint != StorageFactory::READ_HINT_UNBUFFERED
	|| cacheHint == StorageFactory::CACHE_HINT_STORAGE)
      mode &= ~IOFlags::OpenUnbuffered;
    else
      mode |= IOFlags::OpenUnbuffered;

    Storage *file = new RFIOFile(normalise(path), mode);
    if ((cacheHint == StorageFactory::CACHE_HINT_LAZY_DOWNLOAD
	 || cacheHint == StorageFactory::CACHE_HINT_AUTO_DETECT)
	&& ! (mode & IOFlags::OpenWrite))
    {
      if (f->accounting())
        file = new StorageAccountProxy(proto, file);
      file = new LocalCacheFile(file);
    }
    return file;
  }

  virtual void stagein (const std::string &proto,
		        const std::string &path)
  {
    std::string npath = normalise(path);
    size_t castor = npath.find("?path=/castor/");
    size_t rest = npath.find("&");
    if (proto != "rfio" || castor == std::string::npos)
      return;

    castor += 6;
    size_t len = (rest == std::string::npos ? rest : rest-castor);
    std::string stagepath(npath, castor, len);

    stage_options opts;
    opts.stage_host = getenv("STAGE_HOST");
    opts.service_class = getenv("STAGE_SVCCLASS");
    opts.stage_port = 0;
    opts.stage_version = 2;

    stage_prepareToGet_filereq req;
    req.protocol = (char *) "rfio";
    req.filename = (char *) stagepath.c_str();
    req.priority = 0;

    int nresp = 0;
    stage_prepareToGet_fileresp *resp = 0;
    int rc = stage_prepareToGet(0, &req, 1, &resp, &nresp, 0, &opts);
    if (rc < 0)
      throw cms::Exception("RFIOStorageMaker::stagein()")
	<< "Error while staging in '" << stagepath
        << "', error was: " << rfio_serror()
        << " (serrno=" << serrno << ")";

    if (nresp == 1 && resp->errorCode != 0)
      throw cms::Exception("RFIOStorageMaker::stagein()")
	<< "Error while staging in '" << stagepath
        << "', stagein error was: " << resp->errorMessage
        << " (code=" << resp->errorCode << ")";
      
    free(resp->filename);
    free(resp->errorMessage);
    free(resp);
  }

  virtual bool check (const std::string &proto,
		      const std::string &path,
		      IOOffset *size = 0)
  {
    std::string npath = normalise(path);
    if (rfio_access(npath.c_str (), R_OK) != 0)
      return false;

    if (size)
    {
      struct stat buf;
      if (rfio_stat64(npath.c_str (), &buf) != 0)
        return false;

      *size = buf.st_size;
    }

    return true;
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, RFIOStorageMaker, "rfio");
