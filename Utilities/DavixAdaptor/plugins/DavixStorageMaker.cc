#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/DavixAdaptor/interface/DavixFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <unistd.h>
#include <davix.hpp>

class DavixStorageMaker: public StorageMaker
{

public:

  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  virtual std::unique_ptr<Storage> open(const std::string &proto,
			                            const std::string &path,
			                            int mode,
                                        AuxSettings const& aux) const override
  {
    const StorageFactory *f = StorageFactory::get();
    std::string newurl ((proto == "web" ? "http" : proto) + ":" + path);
    auto file = std::make_unique<DavixFile>(newurl, mode);
    return f->wrapNonLocalFile(std::move(file), proto, std::string(), mode);
  }

};

DEFINE_EDM_PLUGIN (StorageMakerFactory, DavixStorageMaker, "http");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DavixStorageMaker, "https");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DavixStorageMaker, "web");
