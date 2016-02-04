#include "Utilities/StorageFactory/interface/IOChannel.h"
#include "Utilities/StorageFactory/src/SysIOChannel.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>
#include <vector>
#include <cassert>

IOSize
IOChannel::read (void *into, IOSize n)
{
  ssize_t s;
  do
    s = ::read (fd (), into, n);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError (edm::errors::FileReadError, "Calling IOChannel::read()", "read()", errno);

  return s;
}

IOSize
IOChannel::readv (IOBuffer *into, IOSize buffers)
{
  assert (! buffers || into);

  // readv may not support zero buffers.
  if (! buffers)
    return 0;

  ssize_t n = 0;

  // Convert the buffers to system format.
  std::vector<iovec> bufs (buffers);
  for (IOSize i = 0; i < buffers; ++i)
  {
    bufs [i].iov_len  = into [i].size ();
    bufs [i].iov_base = (caddr_t) into [i].data ();
  }

  // Read as long as signals cancel the read before doing anything.
  do
    n = ::readv (fd (), &bufs [0], buffers);
  while (n == -1 && errno == EINTR);

  // If it was serious error, throw it.
  if (n == -1)
    throwStorageError (edm::errors::FileReadError, "Calling IOChannel::readv", "readv()", errno);

  // Return the number of bytes actually read.
  return n;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOSize
IOChannel::write (const void *from, IOSize n)
{
  ssize_t s;
  do
    s = ::write (fd (), from, n);
  while (s == -1 && errno == EINTR);

  if (s == -1 && errno != EWOULDBLOCK)
    throwStorageError ("FileWriteError", "Calling IOChannel::write()", "write()", errno);

  return s >= 0 ? s : 0;
}

IOSize
IOChannel::writev (const IOBuffer *from, IOSize buffers)
{
  assert (! buffers || from);

  // writev may not support zero buffers.
  if (! buffers)
    return 0;

  ssize_t n = 0;

  // Convert the buffers to system format.
  std::vector<iovec> bufs (buffers);
  for (IOSize i = 0; i < buffers; ++i)
  {
    bufs [i].iov_len  = from [i].size ();
    bufs [i].iov_base = (caddr_t) from [i].data ();
  }

  // Read as long as signals cancel the read before doing anything.
  do
    n = ::writev (fd (), &bufs [0], buffers);
  while (n == -1 && errno == EINTR);

  // If it was serious error, throw it.
  if (n == -1)
    throwStorageError ("FileWriteError", "Calling IOChannel::writev()", "writev()", errno);

  // Return the number of bytes actually written.
  return n;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void
IOChannel::setBlocking (bool value)
{
#ifdef O_NONBLOCK
  int mode;
  int off = value ? ~0 : ~(O_NDELAY | O_NONBLOCK);
  int on  = value ? O_NONBLOCK : 0;

  if ((mode = fcntl (fd (), F_GETFL, 0)) == -1
      || fcntl (fd (), F_SETFL, (mode & off) | on) == -1)
    throwStorageError ("FileSetBlockingError", "Calling IOChannel::setBlocking()", "fcntl()", errno);
#elif defined FIONBIO
  int mode = value;
  if (ioctl (fd (), FIONBIO, &value) == -1)
    throwStorageError ("FileSetBlockingError", "Calling IOChannel::setBlocking()", "ioctl()", errno);
#endif
}

bool
IOChannel::isBlocking (void) const
{
#ifdef O_NONBLOCK
  int mode;
  if ((mode = fcntl (fd (), F_GETFL, 0)) == -1)
    throwStorageError ("FileIsBlockingError", "Calling IOChannel::isBlocking()", "fcntl()", errno);

  return mode & (O_NDELAY | O_NONBLOCK);
#else // ! O_NONBLOCK
  return true;
#endif // O_NONBLOCK
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
bool
IOChannel::sysclose (IOFD fd, int *error /* = 0 */)
{
  int ret = ::close (fd);
  if (error) *error = errno;
  return ret != -1;
}
