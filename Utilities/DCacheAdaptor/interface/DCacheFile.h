#ifndef DCACHE_ADAPTOR_DCACHE_FILE_H
#define DCACHE_ADAPTOR_DCACHE_FILE_H

#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include <string>

class DCacheFile : public Storage {
public:
  DCacheFile(void);
  DCacheFile(IOFD fd);
  DCacheFile(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  DCacheFile(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~DCacheFile(void) override;

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
  IOFD m_fd;
  bool m_close;
  std::string m_name;
};

#endif  // DCACHE_ADAPTOR_DCACHE_FILE_H
