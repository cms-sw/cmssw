#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/src/SysFile.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include <cassert>

using namespace IOFlags;

IOFD
File::sysduplicate (IOFD fd)
{
  IOFD copyfd;
  if ((copyfd = ::dup (fd)) == EDM_IOFD_INVALID)
    throwStorageError ("File::sysduplicate()", "dup()", errno);

  return copyfd;
}

void
File::sysopen (const char *name, int flags, int perms,
               IOFD &newfd, unsigned int& /*newflags*/)
{
  // Translate our flags to system flags.
  int openflags = 0;

  if ((flags & OpenRead) && (flags & OpenWrite))
    openflags |= O_RDWR;
  else if (flags & OpenRead)
    openflags |= O_RDONLY;
  else if (flags & OpenWrite)
    openflags |= O_WRONLY;

  if (flags & OpenNonBlock)
    openflags |= O_NONBLOCK;

  if (flags & OpenAppend)
    openflags |= O_APPEND;

#ifdef O_SYNC
  if (flags & OpenUnbuffered)
    openflags |= O_SYNC;
#else
  if (flags & OpenUnbuffered)
    newflags |= OpenUnbuffered;
#endif

  if (flags & OpenCreate)
    openflags |= O_CREAT;

  if (flags & OpenExclusive)
    openflags |= O_EXCL;

  if (flags & OpenTruncate)
    openflags |= O_TRUNC;

  if (flags & OpenNotCTTY)
    openflags |= O_NOCTTY;

  if ((newfd = ::open (name, openflags, perms)) == -1)
    throwStorageError ("File::sysopen()", "open()", errno);
}

IOSize
File::read (void *into, IOSize n, IOOffset pos)
{
  assert (pos >= 0);

  ssize_t s;
  do
    s = ::pread (fd (), into, n, pos);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError("File::read()", "pread()", errno);

  return s;
}

IOSize
File::write (const void *from, IOSize n, IOOffset pos)
{
  assert (pos >= 0);

  ssize_t s;
  do
    s = ::pwrite (fd (), from, n, pos);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError("File::write()", "pwrite()", errno);

  if (m_flags & OpenUnbuffered)
    // FIXME: Exception handling?
    flush ();

  return s;
}

IOOffset
File::size (void) const
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);

  struct stat info;
  if (fstat (fd, &info) == -1)
    throwStorageError("File::size()", "fstat()", errno);

  return info.st_size;
}

IOOffset
File::position (IOOffset offset, Relative whence /* = SET */)
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);
  assert (whence == CURRENT || whence == SET || whence == END);

  IOOffset result;
  int      mywhence = (whence == SET ? SEEK_SET
		       : whence == CURRENT ? SEEK_CUR
		       : SEEK_END);
  if ((result = ::lseek (fd, offset, mywhence)) == -1)
    throwStorageError("File::position()", "lseek()", errno);

  return result;
}

void
File::resize (IOOffset size)
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);

  if (ftruncate (fd, size) == -1)
    throwStorageError("File::resize()", "ftruncate()", errno);
}

void
File::flush (void)
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);

#if _POSIX_SYNCHRONIZED_IO > 0
  if (fdatasync (fd) == -1)
    throwStorageError("File::flush()", "fdatasync()", errno);
#elif _POSIX_FSYNC > 0
  if (fsync (fd) == -1)
    throwStorageError("File::flush()", "fsync()", errno);
#endif
}

bool
File::sysclose (IOFD fd, int *error /* = 0 */)
{
  int ret = ::close (fd);
  if (error) *error = errno;
  return ret != -1;
}
