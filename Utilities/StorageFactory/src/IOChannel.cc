#include "Utilities/StorageFactory/interface/IOChannel.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include <algorithm>
#include <cassert>

/** @fn IOSize IOChannel::read (void *into, IOSize n)
    Read at most @a n bytes from the channel into the buffer @a into.

    @return The number of bytes actually read into the buffer.  This
    is always less or equal to @a n.  It can be less if there is
    limited amount of input currently available; zero means that the
    end of file has been reached.  For a connected channel like a
    socket or pipe this indicates the remote end has closed the
    connection.  If the channel is in non-blocking mode and no input
    is currently available, an exception is thrown (FIXME: make this
    simpler; clarify which exception?).  */

/** @fn IOSize IOChannel::readv (IOBuffer *into, IOSize buffers)
    Read into scattered buffers.

    This operation may ignore errors.  If some data are already read
    and an error occurs, the call returns the number of bytes read up
    to that point, hiding the error.  It is assumed that a subsequent
    read will discover persistent errors and that sporadic errors such
    as indication that the read would block can be ignored.

    The operation always fills a buffer completely before proceeding
    to the next one.  The call is handled by the operating system if
    possible; the fall back is to use the single #read() repeatedly.  */

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/** @fn IOSize IOChannel::write (const void *from, IOSize n)
    Write @a n bytes from the buffer at @a from.

    @return The number of bytes actually written.  This is always less
    or equal to the size of the buffer (@a n).  It can be less if the
    channel is unable to accept some of the output.  This can happen
    among others if the channel is in non-blocking mode, but also for
    other implementation and platform-dependent reasons.  */

/** @fn IOSize IOChannel::writev (const IOBuffer *from, IOSize buffers)
    Write from a scattered buffer.

    This operation may ignore errors.  If some data is already written
    and an error occurs, the call returns the number of bytes written
    up to that point, hiding the error.  It is assumed that a
    subsequent write will discover persistent errors.

    Always writes a complete buffer before proceeding to the next one.
    The call is delegated to the operating system if possible.  If the
    system does not support this, falls back on multiple calls to
    single-buffer #write().  */

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

IOChannel::IOChannel (IOFD fd /* = EDM_IOFD_INVALID */)
  : m_fd (fd)
{}

IOChannel::~IOChannel (void)
{}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Get the system file descriptor of the channel.  */
IOFD
IOChannel::fd (void) const
{ return m_fd; }

/** Set the system file descriptor of the channel.  (FIXME: This is
    dangerous.  How to deal with WIN32 flags and state object?)  */
void
IOChannel::fd (IOFD value)
{
  // FIXME: close old one?
  // FIXME: reset state?
  m_fd = value;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Close the channel.  By default closes the underlying operating
    system file descriptor.  */
void
IOChannel::close (void)
{
  if (fd () == EDM_IOFD_INVALID)
    return;

  int error = 0;
  if (! sysclose (fd (), &error))
    throwStorageError ("FileCloseError", "Calling IOChannel::close()",
                       "sysclose()", error);

  fd (EDM_IOFD_INVALID);
}
