#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "Utilities/StorageFactory/interface/StorageAccount.h"
# include "SealBase/Storage.h"
# include <string>

#include<vector>
extern "C" {
  struct iovec64;
}
typedef std::vector<iovec64> IOVec;

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

/** Proxy class that wraps SEAL's #Storage class with one that ticks
    #StorageAccount counters for significant operations.  The returned
    #Storage objects from #StorageMaker are automatically wrapped with
    this class.

    Future improvement would be to implement more methods so that the
    wrapper itself doesn't cause peroformance degradation if the base
    storage does actually implement "sophisticated" features.  */
class StorageAccountProxy : public seal::Storage
{
public:
    StorageAccountProxy (const std::string &storageClass, seal::Storage *baseStorage);
    ~StorageAccountProxy (void);

    using Storage::read;
    using Storage::write;

    virtual seal::IOSize	read (void *into, seal::IOSize n);
    virtual seal::IOSize	write (const void *from, seal::IOSize n);

    virtual seal::IOOffset	position (seal::IOOffset offset, Relative whence = SET);
    virtual void		resize (seal::IOOffset size);
    virtual void		flush (void);
    virtual void		close (void);
   
  virtual void          preseek(const IOVec& iov);

protected:
    std::string			m_storageClass;
    seal::Storage		*m_baseStorage;

    StorageAccount::Counter	&m_statsRead;
    StorageAccount::Counter	&m_statsWrite;
    StorageAccount::Counter	&m_statsPosition;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_PROXY_H
