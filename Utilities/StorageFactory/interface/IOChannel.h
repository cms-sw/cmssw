#ifndef STORAGE_FACTORY_IO_CHANNEL_H
#define STORAGE_FACTORY_IO_CHANNEL_H

#include "Utilities/StorageFactory/interface/IOInput.h"
#include "Utilities/StorageFactory/interface/IOOutput.h"

namespace edm::storage {

  /** Base class for stream-oriented I/O sources and targets, based
    on the operating system file descriptor. */
  class IOChannel : public IOInput, public IOOutput {
  public:
    IOChannel(IOFD fd = EDM_IOFD_INVALID);
    ~IOChannel() override;
    // implicit copy constructor
    // implicit assignment operator

    using IOInput::read;
    using IOOutput::write;

    IOSize read(void *into, IOSize n) override;
    IOSize readv(IOBuffer *into, IOSize buffers) override;

    IOSize write(const void *from, IOSize n) override;
    IOSize writev(const IOBuffer *from, IOSize buffers) override;

    IOFD fd() const;
    void fd(IOFD value);  // FIXME: dangerous?

    void setBlocking(bool value);
    bool isBlocking() const;

  private:
    IOFD m_fd; /*< System file descriptor. */
  };
}  // namespace edm::storage
#endif  // STORAGE_FACTORY_IO_CHANNEL_H
