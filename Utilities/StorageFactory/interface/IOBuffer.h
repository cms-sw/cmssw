#ifndef STORAGE_FACTORY_IO_BUFFER_H
# define STORAGE_FACTORY_IO_BUFFER_H

# include "Utilities/StorageFactory/interface/IOTypes.h"

/** Buffer for I/O operations. */
class IOBuffer
{
public:
  IOBuffer (void);
  IOBuffer (void *data, IOSize length);
  IOBuffer (const void *data, IOSize length);

  void *	data (void) const;
  IOSize	size (void) const;

private:
  void		*m_data;	//< Data
  IOSize	m_length;	//< Length of data in bytes.
};

/** Construct a null I/O buffer.  */
inline
IOBuffer::IOBuffer (void)
  : m_data (nullptr),
    m_length (0)
{}

/** Construct a I/O buffer for reading.  */
inline
IOBuffer::IOBuffer (void *data, IOSize length)
  : m_data (data),
    m_length (length)
{}

/** Construct a I/O buffer for writing.  */
inline
IOBuffer::IOBuffer (const void *data, IOSize length)
  : m_data (const_cast<void *> (data)),
    m_length (length)
{}

/** Return a pointer to the beginning of the buffer's data area.  */
inline void *
IOBuffer::data (void) const
{ return m_data; }

/** Return the buffer's size.  */
inline IOSize
IOBuffer::size (void) const
{ return m_length; }

#endif // STORAGE_FACTORY_IO_BUFFER_H
