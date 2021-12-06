#ifndef STORAGE_FACTORY_FILE_H
#define STORAGE_FACTORY_FILE_H

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/IOFlags.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <string>

namespace edm::storage {

  /** Basic file-related functions.  Nicked from SEAL.  */
  class File : public Storage {
  public:
    File();
    File(IOFD fd, bool autoclose = true);
    File(const char *name, int flags = IOFlags::OpenRead, int perms = 0666);
    File(const std::string &name, int flags = IOFlags::OpenRead, int perms = 0666);
    ~File() override;
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

    IOFD fd() const { return m_fd; }

    bool prefetch(const IOPosBuffer *what, IOSize n) override;
    IOSize read(void *into, IOSize n) override;
    IOSize read(void *into, IOSize n, IOOffset pos) override;
    IOSize readv(IOBuffer *into, IOSize length) override;

    IOSize write(const void *from, IOSize n) override;
    IOSize write(const void *from, IOSize n, IOOffset pos) override;
    IOSize writev(const IOBuffer *from, IOSize length) override;

    IOOffset size() const override;
    IOOffset position(IOOffset offset, Relative whence = SET) override;

    void resize(IOOffset size) override;

    void flush() override;
    void close() override;
    virtual void abort();

    virtual void setAutoClose(bool closeit);

  private:
    enum { InternalAutoClose = 4096 };  //< Close on delete

    File(IOFD fd, unsigned flags);

    IOSize syswrite(const void *from, IOSize n);
    IOSize syswritev(const IOBuffer *from, IOSize length);

    File *duplicate(bool copy) const;
    File *duplicate(File *child) const;
    static IOFD sysduplicate(IOFD fd);
    static void sysopen(const char *name, int flags, int perms, IOFD &newfd, unsigned &newflags);
    static bool sysclose(IOFD fd, int *error = nullptr);

    IOFD m_fd = EDM_IOFD_INVALID; /*< System file descriptor. */
    unsigned m_flags;
  };
}  // namespace edm::storage
#endif  // STORAGE_FACTORY_FILE_H
