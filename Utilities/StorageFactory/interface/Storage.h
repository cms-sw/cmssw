#ifndef STORAGE_FACTORY_STORAGE_H
#define STORAGE_FACTORY_STORAGE_H

#include "Utilities/StorageFactory/interface/IOPosBuffer.h"
#include "Utilities/StorageFactory/interface/IOBuffer.h"

//
// ROOT will probe for prefetching support by calling
// ReadBufferAsync(0, 0)
// Storage turns this into:
// prefetch(0, PREFETCH_PROBE_LENGTH)
//
// For example, if the Storage implementation wants to provide a prefetch
// implementation, but prefers it not to be used by default, it
// should detect the probe and return true.
//
namespace edm::storage {
  constexpr int PREFETCH_PROBE_LENGTH = 4096;

  class Storage {
  public:
    enum Relative { SET, CURRENT, END };

    Storage();

    // undefined, no semantics
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;

    virtual ~Storage();

    int read();
    IOSize read(IOBuffer into);
    virtual IOSize read(void *into, IOSize n) = 0;
    virtual IOSize readv(IOBuffer *into, IOSize buffers);

    IOSize xread(IOBuffer into);
    IOSize xread(void *into, IOSize n);
    IOSize xreadv(IOBuffer *into, IOSize buffers);

    IOSize write(unsigned char byte);
    IOSize write(IOBuffer from);
    virtual IOSize write(const void *from, IOSize n) = 0;
    virtual IOSize writev(const IOBuffer *from, IOSize buffers);

    IOSize xwrite(const void *from, IOSize n);
    IOSize xwrite(IOBuffer from);
    IOSize xwritev(const IOBuffer *from, IOSize buffers);

    virtual bool prefetch(const IOPosBuffer *what, IOSize n);
    virtual IOSize read(void *into, IOSize n, IOOffset pos);
    IOSize read(IOBuffer into, IOOffset pos);
    virtual IOSize readv(IOPosBuffer *into, IOSize buffers);
    virtual IOSize write(const void *from, IOSize n, IOOffset pos);
    IOSize write(IOBuffer from, IOOffset pos);
    virtual IOSize writev(const IOPosBuffer *from, IOSize buffers);

    virtual bool eof() const;
    virtual IOOffset size() const;
    virtual IOOffset position() const;
    virtual IOOffset position(IOOffset offset, Relative whence = SET) = 0;

    virtual void rewind();

    virtual void resize(IOOffset size) = 0;

    virtual void flush();
    virtual void close();
  };
}  // namespace edm::storage
#endif  // STORAGE_FACTORY_STORAGE_H
