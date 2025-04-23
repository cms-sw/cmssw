#ifndef Utilities_StorageFactory_StorageProxyMaker_h
#define Utilities_StorageFactory_StorageProxyMaker_h

#include <memory>

namespace edm::storage {
  class Storage;

  // Base class for makers of generic Storage proxies
  class StorageProxyMaker {
  public:
    StorageProxyMaker() = default;
    virtual ~StorageProxyMaker();

    virtual std::unique_ptr<Storage> wrap(std::unique_ptr<Storage> storage) = 0;
  };
}  // namespace edm::storage

#endif
