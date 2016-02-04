#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/RemoteFile.h"

class HttpStorageMaker : public StorageMaker
{
public:
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode)
  {
    std::string    temp;
    StorageFactory *f = StorageFactory::get();
    int            localfd = RemoteFile::local (f->tempDir(), temp);
    std::string    newurl ((proto == "web" ? "http" : proto) + ":" + path);
    const char     *curlopts [] = {
      "curl", "-L", "-f", "-o", temp.c_str(), "-q", "-s", "--url",
      newurl.c_str (), 0
    };

    return RemoteFile::get (localfd, temp, (char **) curlopts, mode);
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "http");
DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "ftp");
DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "web");
