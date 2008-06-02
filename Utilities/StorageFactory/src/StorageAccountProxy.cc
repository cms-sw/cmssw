#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"

StorageAccountProxy::StorageAccountProxy (const std::string &storageClass,
					  Storage *baseStorage)
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

IOSize
StorageAccountProxy::read (void *into, IOSize n)
{
  StorageAccount::Stamp stats (m_statsRead);
  IOSize result = m_baseStorage->read (into, n);
  stats.tick (result);
  return result;
}

IOSize
StorageAccountProxy::write (const void *from, IOSize n)
{
  StorageAccount::Stamp stats (m_statsWrite);
  IOSize result = m_baseStorage->write (from, n);
  stats.tick (result);
  return result;
}

IOOffset
StorageAccountProxy::position (IOOffset offset, Relative whence)
{
  StorageAccount::Stamp stats (m_statsPosition);
  IOOffset result = m_baseStorage->position (offset, whence);
  stats.tick ();
  return result;
}

void
StorageAccountProxy::resize (IOOffset size)
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

void StorageAccountProxy::preseek (const IOBuffer *offsets, IOSize buffers)
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "preseek"));
  m_baseStorage->preseek (offsets, buffers);
  stats.tick ();
}
