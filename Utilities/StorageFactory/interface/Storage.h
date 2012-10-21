#ifndef STORAGE_FACTORY_STORAGE_H
# define STORAGE_FACTORY_STORAGE_H

# include "Utilities/StorageFactory/interface/IOInput.h"
# include "Utilities/StorageFactory/interface/IOOutput.h"
# include "Utilities/StorageFactory/interface/IOPosBuffer.h"

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
# define PREFETCH_PROBE_LENGTH 4096

class Storage : public virtual IOInput, public virtual IOOutput
{
public:
  enum Relative { SET, CURRENT, END };

  Storage (void);
  virtual ~Storage (void);

  using IOInput::read;
  using IOInput::readv;
  using IOOutput::write;
  using IOOutput::writev;

  virtual bool		prefetch (const IOPosBuffer *what, IOSize n);
  virtual IOSize	read (void *into, IOSize n, IOOffset pos);
  IOSize		read (IOBuffer into, IOOffset pos);
  virtual IOSize	readv (IOPosBuffer *into, IOSize buffers);
  virtual IOSize	write (const void *from, IOSize n, IOOffset pos);
  IOSize		write (IOBuffer from, IOOffset pos);
  virtual IOSize	writev (const IOPosBuffer *from, IOSize buffers);

  virtual bool		eof (void) const;
  virtual IOOffset	size (void) const;
  virtual IOOffset	position (void) const;
  virtual IOOffset	position (IOOffset offset, Relative whence = SET) = 0;

  virtual void		rewind (void);

  virtual void		resize (IOOffset size) = 0;

  virtual void		flush (void);
  virtual void		close (void);

private:
  // undefined, no semantics
  Storage (const Storage &);
  Storage &operator= (const Storage &);
};

#endif // STORAGE_FACTORY_STORAGE_H
