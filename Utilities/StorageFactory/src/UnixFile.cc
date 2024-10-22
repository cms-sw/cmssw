#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/src/SysFile.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cassert>
#include <vector>

using namespace edm::storage;

using namespace IOFlags;

IOFD File::sysduplicate(IOFD fd) {
  IOFD copyfd;
  if ((copyfd = ::dup(fd)) == EDM_IOFD_INVALID)
    throwStorageError("FileDuplicateError", "Calling File::sysduplicate()", "dup()", errno);

  return copyfd;
}

void File::sysopen(const char *name, int flags, int perms, IOFD &newfd, unsigned int & /*newflags*/) {
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

  if ((newfd = ::open(name, openflags, perms)) == -1)
    throwStorageError(edm::errors::FileOpenError, "Calling File::sysopen()", "open()", errno);
}

IOSize File::read(void *into, IOSize n) {
  ssize_t s;
  do
    s = ::read(fd(), into, n);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError(edm::errors::FileReadError, "Calling File::read()", "read()", errno);

  return s;
}

IOSize File::readv(IOBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // readv may not support zero buffers.
  if (!buffers)
    return 0;

  ssize_t n = 0;

  // Convert the buffers to system format.
  std::vector<iovec> bufs(buffers);
  for (IOSize i = 0; i < buffers; ++i) {
    bufs[i].iov_len = into[i].size();
    bufs[i].iov_base = (caddr_t)into[i].data();
  }

  // Read as long as signals cancel the read before doing anything.
  do
    n = ::readv(fd(), &bufs[0], buffers);
  while (n == -1 && errno == EINTR);

  // If it was serious error, throw it.
  if (n == -1)
    throwStorageError(edm::errors::FileReadError, "Calling File::readv", "readv()", errno);

  // Return the number of bytes actually read.
  return n;
}

IOSize File::read(void *into, IOSize n, IOOffset pos) {
  assert(pos >= 0);

  ssize_t s;
  do
    s = ::pread(fd(), into, n, pos);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError(edm::errors::FileReadError, "Calling File::read()", "pread()", errno);

  return s;
}

IOSize File::syswrite(const void *from, IOSize n) {
  ssize_t s;
  do
    s = ::write(fd(), from, n);
  while (s == -1 && errno == EINTR);

  if (s == -1 && errno != EWOULDBLOCK)
    throwStorageError(edm::errors::FileWriteError, "Calling File::syswrite()", "syswrite()", errno);

  return s >= 0 ? s : 0;
}

IOSize File::syswritev(const IOBuffer *from, IOSize buffers) {
  assert(!buffers || from);

  // writev may not support zero buffers.
  if (!buffers)
    return 0;

  ssize_t n = 0;

  // Convert the buffers to system format.
  std::vector<iovec> bufs(buffers);
  for (IOSize i = 0; i < buffers; ++i) {
    bufs[i].iov_len = from[i].size();
    bufs[i].iov_base = (caddr_t)from[i].data();
  }

  // Read as long as signals cancel the read before doing anything.
  do
    n = ::writev(fd(), &bufs[0], buffers);
  while (n == -1 && errno == EINTR);

  // If it was serious error, throw it.
  if (n == -1)
    throwStorageError(edm::errors::FileWriteError, "Calling Fike::syswritev()", "syswritev()", errno);

  // Return the number of bytes actually written.
  return n;
}

IOSize File::write(const void *from, IOSize n, IOOffset pos) {
  assert(pos >= 0);

  ssize_t s;
  do
    s = ::pwrite(fd(), from, n, pos);
  while (s == -1 && errno == EINTR);

  if (s == -1)
    throwStorageError(edm::errors::FileWriteError, "Calling File::write()", "pwrite()", errno);

  if (m_flags & OpenUnbuffered)
    // FIXME: Exception handling?
    flush();

  return s;
}

IOOffset File::size() const {
  IOFD fd = m_fd;
  assert(fd != EDM_IOFD_INVALID);

  struct stat info;
  if (fstat(fd, &info) == -1)
    throwStorageError("FileSizeError", "Calling File::size()", "fstat()", errno);

  return info.st_size;
}

IOOffset File::position(IOOffset offset, Relative whence /* = SET */) {
  IOFD fd = m_fd;
  assert(fd != EDM_IOFD_INVALID);
  assert(whence == CURRENT || whence == SET || whence == END);

  IOOffset result;
  int mywhence = (whence == SET ? SEEK_SET : whence == CURRENT ? SEEK_CUR : SEEK_END);
  if ((result = ::lseek(fd, offset, mywhence)) == -1)
    throwStorageError("FilePositionError", "Calling File::position()", "lseek()", errno);

  return result;
}

void File::resize(IOOffset size) {
  IOFD fd = m_fd;
  assert(fd != EDM_IOFD_INVALID);

  if (ftruncate(fd, size) == -1)
    throwStorageError("FileResizeError", "Calling File::resize()", "ftruncate()", errno);
}

void File::flush() {
  IOFD fd = m_fd;
  assert(fd != EDM_IOFD_INVALID);

#if _POSIX_SYNCHRONIZED_IO > 0
  if (fdatasync(fd) == -1)
    throwStorageError("FileFlushError", "Calling File::flush()", "fdatasync()", errno);
#elif _POSIX_FSYNC > 0
  if (fsync(fd) == -1)
    throwStorageError("FileFlushError", "Calling File::flush()", "fsync()", errno);
#endif
}

bool File::sysclose(IOFD fd, int *error /* = 0 */) {
  int ret = ::close(fd);
  if (error)
    *error = errno;
  return ret != -1;
}
