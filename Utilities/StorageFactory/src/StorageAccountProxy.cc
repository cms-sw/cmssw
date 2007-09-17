//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

StorageAccountProxy::StorageAccountProxy (const std::string &storageClass,
					  seal::Storage *baseStorage)
    : m_storageClass (storageClass),
      m_baseStorage (baseStorage),
      m_statsRead (StorageAccount::counter (m_storageClass, "read")),
      m_statsWrite (StorageAccount::counter (m_storageClass, "write")),
      m_statsPosition (StorageAccount::counter (m_storageClass, "position"))
{
  //  StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "construct"));
  //  stats.tick ();
}

StorageAccountProxy::~StorageAccountProxy (void)
{
    StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "destruct"));
    delete m_baseStorage;
    stats.tick ();
}

seal::IOSize
StorageAccountProxy::read (void *into, seal::IOSize n)
{
    StorageAccount::Stamp stats (m_statsRead);
    seal::IOSize result = m_baseStorage->read (into, n);
    stats.tick (result);
    return result;
}

seal::IOSize
StorageAccountProxy::write (const void *from, seal::IOSize n)
{
    StorageAccount::Stamp stats (m_statsWrite);
    seal::IOSize result = m_baseStorage->write (from, n);
    stats.tick (result);
    return result;
}

seal::IOOffset
StorageAccountProxy::position (seal::IOOffset offset, Relative whence)
{
    StorageAccount::Stamp stats (m_statsPosition);
    seal::IOOffset result = m_baseStorage->position (offset, whence);
    stats.tick ();
    return result;
}

void
StorageAccountProxy::resize (seal::IOOffset size)
{
    StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "resize"));
    m_baseStorage->resize (size);
    stats.tick ();
}

void
StorageAccountProxy::flush (void)
{
    StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "flush"));
    m_baseStorage->flush ();
    stats.tick ();
}

void
StorageAccountProxy::close (void)
{
    StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "close"));
    m_baseStorage->close ();
    stats.tick ();
}


#include "Utilities/StorageTypes/interface/RFIOStorage.h"

void  StorageAccountProxy::preseek(const IOVec& iov) {

  RFIOStorage * rf = 
    dynamic_cast<RFIOStorage*>( m_baseStorage);
  if (rf) {
    StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "preseek"));
    (*rf).preseek(iov);
    stats.tick ();
  }
  
}
