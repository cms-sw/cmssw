#ifndef STORAGE_FACTORY_IO_CHANNEL_H
# define STORAGE_FACTORY_IO_CHANNEL_H

# include "Utilities/StorageFactory/interface/IOInput.h"
# include "Utilities/StorageFactory/interface/IOOutput.h"

/** Base class for stream-oriented I/O sources and targets, based
    on the operating system file descriptor. */
class IOChannel : public virtual IOInput, public virtual IOOutput
{
public:
  IOChannel (IOFD fd = EDM_IOFD_INVALID);
  virtual ~IOChannel (void);
  // implicit copy constructor
  // implicit assignment operator

  using IOInput::read;
  using IOOutput::write;
  
  virtual IOSize	read (void *into, IOSize n);
  virtual IOSize	readv (IOBuffer *into, IOSize buffers);

  virtual IOSize	write (const void *from, IOSize n);
  virtual IOSize	writev (const IOBuffer *from, IOSize buffers);

  virtual IOFD		fd (void) const;
  virtual void		fd (IOFD value); // FIXME: dangerous?

  virtual void		close (void);

  virtual void		setBlocking (bool value);
  virtual bool		isBlocking (void) const;

protected:
  // System implementation
  bool			sysclose (IOFD fd, int *error = 0);

private:
  IOFD			m_fd;		/*< System file descriptor. */
};

#endif // STORAGE_FACTORY_IO_CHANNEL_H
