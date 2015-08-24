#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"

StorageAccountProxy::StorageAccountProxy (const std::string &storageClass,
                                          std::unique_ptr<Storage> baseStorage)
  : m_baseStorage (std::move(baseStorage)),
    m_token(StorageAccount::tokenForStorageClassName(storageClass)),
    m_statsRead (StorageAccount::counter (m_token, StorageAccount::Operation::read)),
    m_statsReadV (StorageAccount::counter (m_token, StorageAccount::Operation::readv)),
    m_statsWrite (StorageAccount::counter (m_token, StorageAccount::Operation::write)),
    m_statsWriteV (StorageAccount::counter (m_token, StorageAccount::Operation::writev)),
    m_statsPosition (StorageAccount::counter (m_token, StorageAccount::Operation::position)),
    m_statsPrefetch (StorageAccount::counter (m_token, StorageAccount::Operation::prefetch))
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_token, StorageAccount::Operation::construct));
  stats.tick ();
}

StorageAccountProxy::~StorageAccountProxy (void)
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_token, StorageAccount::Operation::destruct));
  m_baseStorage.release();
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
  StorageAccount::Stamp stats (StorageAccount::counter (m_token, StorageAccount::Operation::resize));
  m_baseStorage->resize (size);
  stats.tick ();
}

void
StorageAccountProxy::flush (void)
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_token, StorageAccount::Operation::flush));
  m_baseStorage->flush ();
  stats.tick ();
}

void
StorageAccountProxy::close (void)
{
  StorageAccount::Stamp stats (StorageAccount::counter (m_token, StorageAccount::Operation::close));
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
