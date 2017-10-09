#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/LStoreAdaptor/interface/LStoreFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <unistd.h>
#include <iostream>

class LStoreStorageMaker : public StorageMaker
{
  public:
  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual std::unique_ptr<Storage> open (const std::string &proto,
             const std::string &path,
             int mode,
             const AuxSettings&) const override
  {
	std::string fullpath = proto + ":" + path;
    return std::make_unique<LStoreFile> (fullpath, mode);
  }

/* I don't think this is necessary - Melo
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
*/

  virtual bool check (const std::string &proto,
              const std::string &path,
              const AuxSettings&,
              IOOffset *size = 0) const override
  {
	std::string fullpath = proto + ":" + path;
	try {
		LStoreFile fileObj( fullpath ); // = LStoreFile (fullpath);
		*size = fileObj.position( 0, Storage::END );
	} catch ( cms::Exception & e) {
		return false;
	}
	return true;
  }

};
DEFINE_EDM_PLUGIN (StorageMakerFactory, LStoreStorageMaker, "lstore");
