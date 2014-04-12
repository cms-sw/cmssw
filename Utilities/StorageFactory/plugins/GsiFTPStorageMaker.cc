#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/RemoteFile.h"

class GsiFTPStorageMaker : public StorageMaker
{
public:
  virtual Storage *open (const std::string &proto,
			 const std::string &path,
			 int mode) override
  {
    std::string    temp;
    StorageFactory *f = StorageFactory::get();
    int            localfd = RemoteFile::local (f->tempDir(), temp);
    std::string    lurl = "file://" + temp;
    std::string    newurl ((proto == "sfn" ? "gsiftp" : proto) + ":" + path);
    const char	   *ftpopts [] = { "globus-url-copy", newurl.c_str (), lurl.c_str (), 0 };
    return RemoteFile::get (localfd, temp, (char **) ftpopts, mode);
  }
};

DEFINE_EDM_PLUGIN (StorageMakerFactory, GsiFTPStorageMaker, "gsiftp");
DEFINE_EDM_PLUGIN (StorageMakerFactory, GsiFTPStorageMaker, "sfn");
