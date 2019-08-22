#ifndef DAVIX_ADAPTOR_DAVIX_FILE_H
#define DAVIX_ADAPTOR_DAVIX_FILE_H

#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <davix.hpp>

class DavixFile : public Storage {
public:
  DavixFile(void);
  DavixFile(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  DavixFile(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~DavixFile(void) override;
  static void configureDavixLogLevel();

  virtual void create(const char *name, bool exclusive = false, int perms = 0666);
  virtual void create(const std::string &name, bool exclusive = false, int perms = 0666);
  virtual void open(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  virtual void open(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);

  using Storage::position;
  using Storage::read;
  using Storage::write;

  IOSize read(void *into, IOSize n) override;
  IOSize readv(IOBuffer *into, IOSize buffers) override;
  IOSize readv(IOPosBuffer *into, IOSize buffers) override;
  IOSize write(const void *from, IOSize n) override;
  IOOffset position(IOOffset offset, Relative whence = SET) override;
  void resize(IOOffset size) override;

  void close(void) override;
  virtual void abort(void);

private:
  // Cannot use as C++ smart pointer for Davix_fd
  // Because Davix_fd is not available in C++ header files and
  // sizeof cannot with incomplete types
  Davix_fd *m_fd;
  std::unique_ptr<Davix::DavPosix> m_davixPosix;
  std::string m_name;
};

#endif  // DAVIX_ADAPTOR_DAVIX_FILE_H
