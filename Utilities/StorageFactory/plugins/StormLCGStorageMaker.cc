#define ML_DEBUG 1
#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>

class StormLcgGtStorageMaker : public StorageMaker
{
  /* getTURL: Executes lcg-gt and extracts the physical file path */
  std::string getTURL (const std::string &surl) const
  {
    // PrepareToGet timeout  
    std::string timeout("300");
    if(char *p = getenv("CMS_STORM_LCG_GT_TIMEOUT"))
      timeout = p;

    /* Build the command line:
	-b => no BDII contacted
	-T srmv2 => necessary with -b 
	-t timeout */
    std::string comm("lcg-gt -b -T srmv2 -t " + timeout + " srm:" + surl + " file 2>&1"); 
    LogDebug("StormLCGStorageMaker") << "command: " << comm << std::endl;

    FILE *pipe = popen(comm.c_str(), "r");
    if(! pipe)
      throw cms::Exception("StormLCGStorageMaker")
	<< "failed to execute lcg-gt command: "
	<< comm;

    // Get output
    int ch;
    std::string output;
    while ((ch = getc(pipe)) != EOF)
      output.push_back(ch);
    pclose(pipe);

    LogDebug("StormLCGStorageMaker") << "output: " << output << std::endl;
 
    // Extract TURL if possible.
    size_t start = output.find("file:", 0);
    if (start == std::string::npos)
      throw cms::Exception("StormLCGStorageMaker")
        << "no turl found in command '" << comm << "' output:\n" << output;

    start += 5;
    std::string turl(output, start, output.find_first_of("\n", start) - start); 
    LogDebug("StormLCGStorageMaker") << "file to open: " << turl << std::endl;
    return turl;
  }


public:
  virtual std::unique_ptr<Storage> open (const std::string &proto,
			 const std::string &surl,
			 int mode,
       const AuxSettings& ) const override
  {
    const StorageFactory *f = StorageFactory::get();
    StorageFactory::ReadHint readHint = f->readHint();
    StorageFactory::CacheHint cacheHint = f->cacheHint();

    if (readHint != StorageFactory::READ_HINT_UNBUFFERED
	|| cacheHint == StorageFactory::CACHE_HINT_STORAGE)
      mode &= ~IOFlags::OpenUnbuffered;
    else
      mode |= IOFlags::OpenUnbuffered;

    std::string path = getTURL(surl);
    auto file = std::make_unique<File> (path, mode);
    return f->wrapNonLocalFile (std::move(file), proto, path, mode);
  }

  virtual bool check (const std::string &/*proto*/,
		      const std::string &path,
          const AuxSettings&,
		      IOOffset *size = 0) const override
  {
    struct stat st;
    if (stat (getTURL(path).c_str(), &st) != 0)
      return false;

    if (size)
      *size = st.st_size;

    return true;
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, StormLcgGtStorageMaker, "storm-lcg");
