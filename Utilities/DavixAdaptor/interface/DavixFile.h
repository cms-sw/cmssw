#ifndef DAVIX_ADAPTOR_DAVIX_FILE_H
#define DAVIX_ADAPTOR_DAVIX_FILE_H

#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <davix.hpp>

class DavixFile : public Storage {
public:
  DavixFile(void);
  DavixFile(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  DavixFile(const std::string &name, int flags = IOFlags::OpenRead,
            int perms = 0666);
  ~DavixFile(void);
  static Davix::Context *getDavixInstance();

  virtual void create(const char *name, bool exclusive = false,
                      int perms = 0666);
  virtual void create(const std::string &name, bool exclusive = false,
                      int perms = 0666);
  virtual void open(const char *name, int flags = IOFlags::OpenRead,
                    int perms = 0666);
  virtual void open(const std::string &name, int flags = IOFlags::OpenRead,
                    int perms = 0666);

  using Storage::read;
  using Storage::write;
  using Storage::position;

  virtual IOSize read(void *into, IOSize n);
  virtual IOSize write(const void *from, IOSize n);
  virtual IOOffset position(IOOffset offset, Relative whence = SET);
  virtual void resize(IOOffset size);

  virtual void close(void);
  virtual void abort(void);

private:
  Davix_fd *m_fd;
  Davix::Context *davixContext;
  Davix::DavPosix *davixPosix;
  std::string m_name;
};

#endif // DAVIX_ADAPTOR_DAVIX_FILE_H
