#ifndef STORAGE_FACTORY_FILE_H
#define STORAGE_FACTORY_FILE_H

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/IOChannel.h"
#include <string>

/** Basic file-related functions.  Nicked from SEAL.  */
class File : public IOChannel, public Storage {
public:
  File(void);
  File(IOFD fd, bool autoclose = true);
  File(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  File(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  ~File(void) override;
  // implicit copy constructor
  // implicit assignment operator

  virtual void create(const char *name, bool exclusive = false, int perms = 0666);
  virtual void create(const std::string &name, bool exclusive = false, int perms = 0666);
  virtual void open(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
  virtual void open(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
  virtual void attach(IOFD fd);

  using Storage::position;
  using Storage::read;
  using Storage::readv;
  using Storage::write;
  using Storage::writev;

  bool prefetch(const IOPosBuffer *what, IOSize n) override;
  IOSize read(void *into, IOSize n) override;
  IOSize read(void *into, IOSize n, IOOffset pos) override;
  IOSize readv(IOBuffer *into, IOSize length) override;

  IOSize write(const void *from, IOSize n) override;
  IOSize write(const void *from, IOSize n, IOOffset pos) override;
  IOSize writev(const IOBuffer *from, IOSize length) override;

  IOOffset size(void) const override;
  IOOffset position(IOOffset offset, Relative whence = SET) override;

  void resize(IOOffset size) override;

  void flush(void) override;
  void close(void) override;
  virtual void abort(void);

  virtual void setAutoClose(bool closeit);

private:
  enum { InternalAutoClose = 4096 };  //< Close on delete

  File(IOFD fd, unsigned flags);

  File *duplicate(bool copy) const;
  File *duplicate(File *child) const;
  static IOFD sysduplicate(IOFD fd);
  static void sysopen(const char *name, int flags, int perms, IOFD &newfd, unsigned &newflags);
  static bool sysclose(IOFD fd, int *error = nullptr);

  unsigned m_flags;
};

#endif  // STORAGE_FACTORY_FILE_H
