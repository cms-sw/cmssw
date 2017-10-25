#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/RemoteFile.h"

class HttpStorageMaker : public StorageMaker
{
public:
  std::unique_ptr<Storage> open (const std::string &proto,
			 const std::string &path,
			 int mode,
       const AuxSettings&) const override
  {
    std::string    temp;
    const StorageFactory *f = StorageFactory::get();
    int            localfd = RemoteFile::local (f->tempDir(), temp);
    std::string    newurl ((proto == "web" ? "http" : proto) + ":" + path);
    const char     *curlopts [] = {
      "curl", "-L", "-f", "-o", temp.c_str(), "-q", "-s", "--url",
      newurl.c_str (), nullptr
    };

    return RemoteFile::get (localfd, temp, (char **) curlopts, mode);
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "ftp");
