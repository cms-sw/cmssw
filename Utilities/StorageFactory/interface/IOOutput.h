#ifndef STORAGE_FACTORY_IO_OUTPUT_H
# define STORAGE_FACTORY_IO_OUTPUT_H

# include "Utilities/StorageFactory/interface/IOBuffer.h"

/** Abstract base class for stream-oriented output targets. */
class IOOutput
{
public:
  virtual ~IOOutput (void);
  // implicit constructor
  // implicit copy constructor
  // implicit assignment operator

  IOSize		write (unsigned char byte);
  IOSize		write (IOBuffer from);
  virtual IOSize	write (const void *from, IOSize n) = 0;
  virtual IOSize	writev (const IOBuffer *from, IOSize buffers);

  IOSize		xwrite (const void *from, IOSize n);
  IOSize		xwrite (IOBuffer from);
  IOSize		xwritev (const IOBuffer *from, IOSize buffers);
};

#endif // STORAGE_FACTORY_IO_OUTPUT_H
