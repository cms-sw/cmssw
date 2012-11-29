#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"

StorageAccountProxy::StorageAccountProxy (const std::string &storageClass,
					  Storage *baseStorage)
  : m_storageClass (storageClass),
    m_baseStorage (baseStorage),
    m_statsRead (StorageAccount::counter (m_storageClass, "read")),
    m_statsReadV (StorageAccount::counter (m_storageClass, "readv")),
    m_statsWrite (StorageAccount::counter (m_storageClass, "write")),
    m_statsWriteV (StorageAccount::counter (m_storageClass, "writev")),
    m_statsPosition (StorageAccount::counter (m_storageClass, "position")),
    m_statsPrefetch (StorageAccount::counter (m_storageClass, "prefetch"))
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_storageClass, "construct"));
  stats.tick ();
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
StorageAccountProxy::read (void *into, IOSize n, IOOffset pos)
{
  StorageAccount::Stamp stats (m_statsRead);
  IOSize result = m_baseStorage->read (into, n, pos);
  stats.tick (result);
  return result;
}

IOSize
StorageAccountProxy::readv (IOBuffer *into, IOSize n)
{
  StorageAccount::Stamp stats (m_statsReadV);
  IOSize result = m_baseStorage->readv (into, n);
  stats.tick (result, n);
  return result;
}

IOSize
StorageAccountProxy::readv (IOPosBuffer *into, IOSize n)
{
  StorageAccount::Stamp stats (m_statsReadV);
  IOSize result = m_baseStorage->readv (into, n);
  stats.tick (result, n);
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

IOSize
StorageAccountProxy::write (const void *from, IOSize n, IOOffset pos)
{
  StorageAccount::Stamp stats (m_statsWrite);
  IOSize result = m_baseStorage->write (from, n, pos);
  stats.tick (result);
  return result;
}

IOSize
StorageAccountProxy::writev (const IOBuffer *from, IOSize n)
{
  StorageAccount::Stamp stats (m_statsWriteV);
  IOSize result = m_baseStorage->writev (from, n);
  stats.tick (result, n);
  return result;
}

IOSize
StorageAccountProxy::writev (const IOPosBuffer *from, IOSize n)
{
  StorageAccount::Stamp stats (m_statsWriteV);
  IOSize result = m_baseStorage->writev (from, n);
  stats.tick (result, n);
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

bool
StorageAccountProxy::prefetch (const IOPosBuffer *what, IOSize n)
{
  StorageAccount::Stamp stats (m_statsPrefetch);
  bool value = m_baseStorage->prefetch(what, n);
  if (value)
  {
    IOSize total = 0;
    for (IOSize i = 0; i < n; ++i)
      total += what[i].size();
    stats.tick (total);
  }
  return value;
}
