#include "Utilities/StorageFactory/interface/IOOutput.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <cassert>

/// Destruct the stream.  A no-op.
IOOutput::~IOOutput (void)
{}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** @fn IOSize IOOutput::write (const void *from, IOSize n)
    Write @a n bytes of data starting at address @a from.

    @return The number of bytes written.  Normally this will be @a n,
    but can be less, even zero, for example if the stream is
    non-blocking mode and cannot accept input at this time.

    @throws In case of error, an exception is thrown.  However if the
    stream is in non-blocking mode and cannot accept output, it will
    @em not throw an exception -- the return value will be less than
    requested.  */

/** Write a single byte to the output stream.  This method is simply
    redirected to #write(const void *, IOSize).

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes written.  Normally this will be one,
    but can be zero if the stream is in non-blocking mode and cannot
    accept output at this time.

    @throws In case of error, an exception is thrown.  However if the
    stream is in non-blocking mode and cannot accept output, it will
    @em not throw an exception -- zero will be returned.  */
IOSize
IOOutput::write (unsigned char byte)
{ return write (&byte, 1); }

/** Write to the output stream the buffer @a from.  This method is
    simply redirected to #write(const void *, IOSize).

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes actually written.  This is less or
    equal to the size of the buffer.  The value can be less than
    requested if the stream is unable to accept all the output for
    platform or implementation reasons.

    @throws In case of error, an exception is thrown.  However if the
    stream is in non-blocking mode and cannot accept output, it will
    @em not throw an exception -- the return value will be less than
    requested.  */
IOSize
IOOutput::write (IOBuffer from)
{ return write (from.data (), from.size ()); }

/** Write to the output stream from multiple buffers.  There are @a
    buffers to fill in an array starting at @a from.  The buffers are
    filled in the order given, each buffer fully before the subsequent
    buffers.  The method uses #write(const void *, IOSize), but may be
    implemented more efficiently in derived classes.

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes actually written.  This is less or
    equal to the size of the buffers.  The value can be less than
    requested if the stream is unable to accept all the output for
    platform or implementation reasons.  Note that the return value
    indicates the number of bytes written, not the number of buffers;
    it is the sum total of bytes written from all the buffers.

    @throws In case of error, an exception is thrown.  However if the
    stream is in non-blocking mode and cannot accept output, it will
    @em not throw an exception -- the return value will be less than
    requested.  */
IOSize
IOOutput::writev (const IOBuffer *from, IOSize buffers)
{
  assert (! buffers || from);

  // Keep writing as long as possible; ignore errors if we have
  // written something, otherwise pass it on.
  IOSize x;
  IOSize done = 0;
  try
  {
    for (IOSize i = 0; i < buffers; ++i)
    {
      done += (x = write (from [i].data (), from [i].size ()));
      if (x < from [i].size ())
	break;
    }
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
/** Like the corresponding #write() method but writes until the
    requested number of bytes are written.  Writes @a from contents.
    This method is redirected to #xwrite(const void *, IOSize).

    Unlike #write() which may write less data than requested, this
    function attempts to write, possibly in multiple #write() calls,
    the exact requested amount of data.  It stops writing only in
    case of error.

    If you know the stream blocks on #write() and it would be
    inconvenient to handle partial #write(), use this method as a
    convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking output.

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes actually written from the buffer, i.e.
    the size of the buffer.

    @throws All exceptions from #write() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already written some data.  */
IOSize
IOOutput::xwrite (IOBuffer from)
{ return xwrite (from.data (), from.size ()); }

/** Like the corresponding #write() method but writes until the
    requested number of bytes are written.  Writes data from the
    buffer @a from for its full size.

    Unlike #write() which may write less data than requested, this
    function attempts to write, possibly in multiple #write() calls,
    the exact requested amount of data.  It stops writing only in
    case of error.

    If you know the stream blocks on #write() and it would be
    inconvenient to handle partial #write(), use this method as a
    convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking output.

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes actually written from the buffer, i.e.
    the size of the buffer.

    @throws All exceptions from #write() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already written some data.  */
IOSize
IOOutput::xwrite (const void *from, IOSize n)
{
  // Keep writing until we've written it all.  Let errors fly over.
  IOSize done = 0;
  while (done < n)
    done += write ((const char *) from + done, n - done);

  return done;
}

/** Like the corresponding #writev() method but writes until the
    requested number of bytes are written.  Writes data from @a
    buffers starting at @a from, each for its full size.  The
    buffers are filled in the order given.  This method uses
    #xwrite(const void *, IOSize).

    Unlike #write() which may write less data than requested, this
    function attempts to write, possibly in multiple #write() calls,
    the exact requested amount of data.  It stops writing only in
    case of error.

    If you know the stream blocks on #write() and it would be
    inconvenient to handle partial #write(), use this method as a
    convenience for hiding platforms and circumstance differences.
    It makes no sense to use this method with non-blocking output.

    Note that derived classes should not normally call this method,
    as it simply routes the call back to derived class through the
    other virtual functions.  Use this method only at the "outside
    edge" when transferring calls from one object to another, not
    in up/down calls in the inheritance tree.

    @return The number of bytes actually written from the buffer, i.e.
    the size of the buffer.

    @throws All exceptions from #write() are passed through unhandled.
    Therefore it is possible that an exception is thrown when this
    function has already written some data.  */
IOSize
IOOutput::xwritev (const IOBuffer *from, IOSize buffers)
{
  // Keep writing until we've written it all.  Let errors fly over.
  // FIXME: Use writev(from, buffers) and then sort out in case of
  // failure, the writev probably succeed directly with much less
  // overhead.
  assert (! buffers || from);

  IOSize done = 0;
  for (IOSize i = 0; i < buffers; ++i)
    done += xwrite (from [i].data (), from [i].size ());

  return done;
}
