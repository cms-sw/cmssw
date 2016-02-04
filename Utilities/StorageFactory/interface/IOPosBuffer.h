#ifndef STORAGE_FACTORY_IO_POS_BUFFER_H
# define STORAGE_FACTORY_IO_POS_BUFFER_H

# include "Utilities/StorageFactory/interface/IOTypes.h"

/** Buffer for I/O operations. */
struct IOPosBuffer
{
public:
  IOPosBuffer (void);
  IOPosBuffer (IOOffset offset, void *data, IOSize length);
  IOPosBuffer (IOOffset offset, const void *data, IOSize length);

  IOOffset	offset (void) const;
  void *	data (void) const;
  IOSize	size (void) const;

private:
  IOOffset	m_offset;	//< File offset.
  void		*m_data;	//< Data
  IOSize	m_length;	//< Length of data in bytes.
};

/** Construct a null I/O buffer.  */
inline
IOPosBuffer::IOPosBuffer (void)
  : m_offset (0),
    m_data (0),
    m_length (0)
{}

/** Construct a I/O buffer for reading.  */
inline
IOPosBuffer::IOPosBuffer (IOOffset offset, void *data, IOSize length)
  : m_offset (offset),
    m_data (data),
    m_length (length)
{}

/** Construct a I/O buffer for writing.  */
inline
IOPosBuffer::IOPosBuffer (IOOffset offset, const void *data, IOSize length)
  : m_offset (offset),
    m_data (const_cast<void *> (data)),
    m_length (length)
{}

/** Return the file offset where I/O is expected to occur.  */
inline IOOffset
IOPosBuffer::offset (void) const
{ return m_offset; }

/** Return a pointer to the beginning of the buffer's data area.  */
inline void *
IOPosBuffer::data (void) const
{ return m_data; }

/** Return the buffer's size.  */
inline IOSize
IOPosBuffer::size (void) const
{ return m_length; }

#endif // STORAGE_FACTORY_IO_POS_BUFFER_H
