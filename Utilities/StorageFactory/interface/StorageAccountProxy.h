#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
#define STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H

#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include <string>
#include <memory>

/** Proxy class that wraps SEAL's #Storage class with one that ticks
    #StorageAccount counters for significant operations.  The returned
    #Storage objects from #StorageMaker are automatically wrapped with
    this class.

    Future improvement would be to implement more methods so that the
    wrapper itself doesn't cause peroformance degradation if the base
    storage does actually implement "sophisticated" features.  */
class StorageAccountProxy : public Storage {
public:
  StorageAccountProxy(const std::string &storageClass, std::unique_ptr<Storage> baseStorage);
  ~StorageAccountProxy(void) override;

  using Storage::read;
  using Storage::write;

  bool prefetch(const IOPosBuffer *what, IOSize n) override;
  IOSize read(void *into, IOSize n) override;
  IOSize read(void *into, IOSize n, IOOffset pos) override;
  IOSize readv(IOBuffer *into, IOSize n) override;
  IOSize readv(IOPosBuffer *into, IOSize n) override;
  IOSize write(const void *from, IOSize n) override;
  IOSize write(const void *from, IOSize n, IOOffset pos) override;
  IOSize writev(const IOBuffer *from, IOSize n) override;
  IOSize writev(const IOPosBuffer *from, IOSize n) override;

  IOOffset position(IOOffset offset, Relative whence = SET) override;
  void resize(IOOffset size) override;
  void flush(void) override;
  void close(void) override;

protected:
  void releaseStorage() { get_underlying_safe(m_baseStorage).release(); }

  edm::propagate_const<std::unique_ptr<Storage>> m_baseStorage;

  StorageAccount::StorageClassToken m_token;
  StorageAccount::Counter &m_statsRead;
  StorageAccount::Counter &m_statsReadV;
  StorageAccount::Counter &m_statsWrite;
  StorageAccount::Counter &m_statsWriteV;
  StorageAccount::Counter &m_statsPosition;
  StorageAccount::Counter &m_statsPrefetch;
};

#endif  // STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
