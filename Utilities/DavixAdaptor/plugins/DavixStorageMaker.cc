#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/DavixAdaptor/interface/DavixFile.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include <davix.hpp>
#include <unistd.h>

class DavixStorageMaker : public StorageMaker {
public:
  /** Open a storage object for the given URL (protocol + path), using the
      @a mode bits.  No temporary files are downloaded.  */
  std::unique_ptr<Storage> open(const std::string &proto,
                                const std::string &path,
                                int mode,
                                AuxSettings const &aux) const override {
    const StorageFactory *f = StorageFactory::get();
    std::string newurl((proto == "web" ? "http" : proto) + ":" + path);
    auto file = std::make_unique<DavixFile>(newurl, mode);
    return f->wrapNonLocalFile(std::move(file), proto, std::string(), mode);
  }

  bool check(const std::string &proto,
             const std::string &path,
             const AuxSettings &aux,
             IOOffset *size = nullptr) const override {
    std::string newurl((proto == "web" ? "http" : proto) + ":" + path);
    Davix::DavixError *err = nullptr;
    Davix::Context c;
    Davix::DavPosix davixPosix(&c);
    Davix::StatInfo info;
    davixPosix.stat64(nullptr, newurl, &info, &err);
    if (err) {
      std::unique_ptr<Davix::DavixError> davixErrManaged(err);
      cms::Exception ex("FileCheckError");
      ex << "Check failed with error " << err->getErrMsg().c_str() << " and error code" << err->getStatus();
      ex.addContext("Calling DavixFile::check()");
      throw ex;
    }
    if (size) {
      *size = info.size;
    }
    return true;
  }
};

DEFINE_EDM_PLUGIN(StorageMakerFactory, DavixStorageMaker, "http");
DEFINE_EDM_PLUGIN(StorageMakerFactory, DavixStorageMaker, "https");
DEFINE_EDM_PLUGIN(StorageMakerFactory, DavixStorageMaker, "web");
