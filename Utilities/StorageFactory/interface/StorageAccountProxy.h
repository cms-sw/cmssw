#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H

# include "Utilities/StorageFactory/interface/StorageAccount.h"
# include "Utilities/StorageFactory/interface/Storage.h"
# include "FWCore/Utilities/interface/get_underlying_safe.h"
# include <string>
#include <memory>

/** Proxy class that wraps SEAL's #Storage class with one that ticks
    #StorageAccount counters for significant operations.  The returned
    #Storage objects from #StorageMaker are automatically wrapped with
    this class.

    Future improvement would be to implement more methods so that the
    wrapper itself doesn't cause peroformance degradation if the base
    storage does actually implement "sophisticated" features.  */
class StorageAccountProxy : public Storage
{
public:
  StorageAccountProxy (const std::string &storageClass, std::unique_ptr<Storage> baseStorage);
  ~StorageAccountProxy (void);

  using Storage::read;
  using Storage::write;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	read (void *into, IOSize n, IOOffset pos);
  virtual IOSize	readv (IOBuffer *into, IOSize n);
  virtual IOSize	readv (IOPosBuffer *into, IOSize n);
  virtual IOSize	write (const void *from, IOSize n);
  virtual IOSize	write (const void *from, IOSize n, IOOffset pos);
  virtual IOSize	writev (const IOBuffer *from, IOSize n);
  virtual IOSize	writev (const IOPosBuffer *from, IOSize n);

  virtual IOOffset	position (IOOffset offset, Relative whence = SET);
  virtual void		resize (IOOffset size);
  virtual void		flush (void);
  virtual void		close (void);

protected:
  void releaseStorage() {get_underlying_safe(m_baseStorage).release();}

  edm::propagate_const<std::unique_ptr<Storage>> m_baseStorage;

  StorageAccount::StorageClassToken m_token;
  StorageAccount::Counter &m_statsRead;
  StorageAccount::Counter &m_statsReadV;
  StorageAccount::Counter &m_statsWrite;
  StorageAccount::Counter &m_statsWriteV;
  StorageAccount::Counter &m_statsPosition;
  StorageAccount::Counter &m_statsPrefetch;
};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
