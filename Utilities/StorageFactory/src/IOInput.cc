#include "Utilities/StorageFactory/interface/IOInput.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cassert>

/// Destruct the stream.  A no-op.
IOInput::~IOInput (void)
{}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** @fn IOSize IOInput::read(void *into, IOSize n)

    Read into @a into at most @a n number of bytes.

    If this is a blocking stream, the call will block until some data
    can be read, end of input is reached, or an exception is thrown.
    For a non-blocking stream the available input is returned.  If
    none is available, an exception is thrown.

    @return The number of bytes actually read.  This is less or equal
    to the size of the buffer.  Zero indicates that the end of the
    input has been reached: end of file, or remote end closing for a
    connected channel like a pipe or a socket.  Otherwise the value
    can be less than requested if limited amount of input is currently
    available for platform or implementation reasons.

    @throws In case of error, a #IOError exception is thrown.  This
    includes the situation where the input stream is in non-blocking
    mode and no input is currently available (FIXME: make this
    simpler; clarify which exception).  */

/** Read the next single byte from the input stream and return it as
    an @c unsigned @c char cast to an @c int, -1 to indicate end of
    intput data.

    If this is a blocking stream, the call will block until the byte
    can be read, end of data is reached, or an exception is thrown.
    For a non-blocking input a character is returned if one is
    available, otherwise an exception is thrown.

    The base class implementation simply forwards the call to
    #read(void *, IOSize) method.

    @return The byte cast from a <tt>unsigned char</tt> to an @c int
    (in range 0...255, inclusive) if one could be read, or @c -1 to
    indicate end of input data.

    @throws In case of error, a #IOError exception is thrown.  This
    includes the situation where the input stream is in non-blocking
    mode and no input is currently available (FIXME: make this
    simpler; clarify which exception).  */
int
IOInput::read (void)
{
  unsigned char byte;
  IOSize n = read (&byte, 1);
  return n == 0 ? -1 : byte;
}

/** Read from the input stream into the buffer starting at @a into
    and of size @a n.

    If this is a blocking stream, the call will block until some data
    can be read, end of input is reached, or an exception is thrown.
    For a non-blocking stream the available input is returned.  If
    none is available, an exception is thrown.

    The base class implementation simply forwards the call to
    #read(void *, IOSize) method.

    @return The number of bytes actually read.  This is less or equal
    to the size of the buffer.  Zero indicates that the end of the
    input has been reached: end of file, or remote end closing for a
    connected channel like a pipe or a socket.  Otherwise the value
    can be less than requested if limited amount of input is currently
    available for platform or implementation reasons.

    @throws In case of error, a #IOError exception is thrown.  This
    includes the situation where the input stream is in non-blocking
    mode and no input is currently available (FIXME: make this
    simpler; clarify which exception).  */
IOSize
IOInput::read (IOBuffer into)
{ return read (into.data (), into.size ()); }

/** Read from the input stream into multiple scattered buffers.
    There are @a buffers to fill in an array starting at @a into;
    the memory those buffers occupy does not need to be contiguous.
    The buffers are filled in the order given, eac buffer is filled
    fully before the subsequent buffers.

    If this is a blocking stream, the call will block until some data
    can be read, end of input is reached, or an exception is thrown.
    For a non-blocking stream the available input is returned.  If
    none is available, an exception is thrown.

    The base class implementation uses #read(void *, IOSize) method,
    but derived classes may implement a more efficient alternative.

    @return The number of bytes actually read.  This is less or equal
    to the size of the buffer.  Zero indicates that the end of the
    input has been reached: end of file, or remote end closing for a
    connected channel like a pipe or a socket.  Otherwise the value
    can be less than requested if limited amount of input is currently
    available for platform or implementation reasons.  Note that the
    return value indicates the number of bytes read, not the number of
    buffers; it is the sum total of bytes filled into all the buffers.

    @throws In case of error, a #IOError exception is thrown.  However
    if some data has already been read, the error is swallowed and the
    method returns the data read so far.  It is assumed that
    persistent errors will occur anyway on the next read and sporadic
    errors like stream becoming unvailable can be ignored.  Use
    #xread() if a different policy is desirable.  */
IOSize
IOInput::readv (IOBuffer *into, IOSize buffers)
{
  assert (! buffers || into);

  // Keep reading as long as possible; ignore errors if we have read
  // something, otherwise pass it on.
  IOSize status;
  IOSize done = 0;
  try
  {
    for (IOSize i = 0; i < buffers; done += status, ++i)
      if ((status = read (into [i])) == 0)
	break;
  }
  catch (cms::Exception &)
  {
    if (! done)
      throw;
  }

  return done;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Like the corresponding #read() method but reads until the
    requested number of bytes are read or end of file is reached.
    Reads @a n bytes of data into the buffer @a into.  This method is
    simply redirected to #xread(void *, IOSize).

    Unlike #read() which may return less data than requested, this
    function attempts to read, possibly in multiple #read() calls, the
    exact requested amount of data.  It stops reading only if it
    reaches the end of the input stream (i.e., #read() returns zero).

    If the you know the stream blocks on #read() and it would be
    inconvenient to handle partial #read() results, use this method as
    a convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking input.

    @return The number of bytes actually read into the buffer, i.e.
    the size of the buffer.  It will be less only if the end of the
    file has been reached.

    @throws All exceptions from #read() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already read some data.  */
IOSize
IOInput::xread (IOBuffer into)
{ return xread (into.data (), into.size ()); }

/** Like the corresponding #read() method but reads until the
    requested number of bytes are read or end of file is reached.
    Reads data into the buffer @a into for its full size.

    Unlike #read() which may return less data than requested, this
    function attempts to read, possibly in multiple #read() calls, the
    exact requested amount of data.  It stops reading only if it
    reaches the end of the input stream (i.e., #read() returns zero).

    If the you know the stream blocks on #read() and it would be
    inconvenient to handle partial #read() results, use this method as
    a convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking input.

    @return The number of bytes actually read into the buffer, i.e.
    the size of the buffer.  It will be less only if the end of the
    file has been reached.

    @throws All exceptions from #read() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already read some data.  */
IOSize
IOInput::xread (void *into, IOSize n)
{
  assert (into);

  // Keep reading as long as possible.  Let system errors fly over
  // us, they are a hard error.
  IOSize x;
  IOSize done = 0;
  while (done < n && (x = read ((char *) into + done, n - done)))
    done += x;

  return done;
}

/** Like the corresponding #readv() method but reads until the
    requested number of bytes are read or end of file is reached.
    Reads data into @a buffers starting at @a into, each for its full
    size.  The buffers are filled in the order given.  This method
    uses #xread(void *, IOSize).

    Unlike #readv() which may return less data than requested, this
    function attempts to read, possibly in multiple #read() calls, the
    exact requested amount of data.  It stops reading only if it
    reaches the end of the input stream (i.e., #read() returns zero).

    If the you know the stream blocks on #read() and it would be
    inconvenient to handle partial #read() results, use this method as
    a convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking input.

    @return The number of bytes actually read into the buffer, i.e.
    the size of the buffer.  It will be less only if the end of the
    file has been reached.

    @throws All exceptions from #read() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already read some data.  */
IOSize
IOInput::xreadv (IOBuffer *into, IOSize buffers)
{
  // FIXME: Use read(into, buffers) and then sort out in case of
  // failure, the readv probably succeed directly with much less
  // overhead.

  assert (! buffers || into);

  // Keep reading as long as possible.  Let system errors fly
  // over us, they are a hard error.
  IOSize x;
  IOSize done = 0;
  for (IOSize i = 0; i < buffers; ++i)
  {
    done += (x = xread (into [i]));
    if (x < into [i].size ())
      break;
  }
  return done;
}
