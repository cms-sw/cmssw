#ifndef Utilities_StorageFactory_StorageProxyMaker_h
#define Utilities_StorageFactory_StorageProxyMaker_h

#include <memory>
#include <string>

namespace edm::storage {
  class Storage;

  // Base class for makers of generic Storage proxies
  class StorageProxyMaker {
  public:
    StorageProxyMaker() = default;
    virtual ~StorageProxyMaker();

    virtual std::unique_ptr<Storage> wrap(std::string const& url, std::unique_ptr<Storage> storage) const = 0;
  };
}  // namespace edm::storage

#endif
