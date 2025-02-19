#ifndef STORAGE_FACTORY_IO_INPUT_H
# define STORAGE_FACTORY_IO_INPUT_H

# include "Utilities/StorageFactory/interface/IOBuffer.h"

/** Abstract base class for stream-oriented input sources. */
class IOInput
{
public:
  virtual ~IOInput (void);
  // implicit constructor
  // implicit copy constructor
  // implicit assignment operator

  int			read (void);
  IOSize		read (IOBuffer into);
  virtual IOSize	read (void *into, IOSize n) = 0;
  virtual IOSize	readv (IOBuffer *into, IOSize buffers);

  IOSize		xread (IOBuffer into);
  IOSize		xread (void *into, IOSize n);
  IOSize		xreadv (IOBuffer *into, IOSize buffers);
};

#endif // STORAGE_FACTORY_IO_INPUT_H
