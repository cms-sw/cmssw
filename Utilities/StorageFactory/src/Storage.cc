#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

using namespace edm::storage;

Storage::Storage() {}

Storage::~Storage() {}

//////////////////////////////////////////////////////////////////////
IOSize Storage::read(IOBuffer into, IOOffset pos) { return read(into.data(), into.size(), pos); }

IOSize Storage::read(void *into, IOSize n, IOOffset pos) {
  // FIXME: this is not thread safe!  split into separate interface
  // that a particular storage can choose to support or not?  make
  // sure that throw semantics are correct here!
  // FIXME: use saveposition object in case exceptions are thrown?
  IOOffset here = position();
  position(pos);
  n = read(into, n);
  position(here);
  return n;
}

IOSize Storage::readv(IOPosBuffer *into, IOSize n) {
  IOOffset here = position();
  IOSize total = 0;
  for (IOSize i = 0; i < n; ++i) {
    try {
      position(into[i].offset());
      total += read(into[i].data(), into[i].size());
    } catch (cms::Exception &) {
      if (!total)
        throw;
      break;
    }
  }
  position(here);
  return total;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** @fn IOSize Storage::read(void *into, IOSize n)

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
int Storage::read() {
  unsigned char byte;
  IOSize n = read(&byte, 1);
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
IOSize Storage::read(IOBuffer into) { return read(into.data(), into.size()); }

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
IOSize Storage::readv(IOBuffer *into, IOSize buffers) {
  assert(!buffers || into);

  // Keep reading as long as possible; ignore errors if we have read
  // something, otherwise pass it on.
  IOSize status;
  IOSize done = 0;
  try {
    for (IOSize i = 0; i < buffers; done += status, ++i)
      if ((status = read(into[i])) == 0)
        break;
  } catch (cms::Exception &) {
    if (!done)
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
IOSize Storage::xread(IOBuffer into) { return xread(into.data(), into.size()); }

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
IOSize Storage::xread(void *into, IOSize n) {
  assert(into);

  // Keep reading as long as possible.  Let system errors fly over
  // us, they are a hard error.
  IOSize x;
  IOSize done = 0;
  while (done < n && (x = read((char *)into + done, n - done)))
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
IOSize Storage::xreadv(IOBuffer *into, IOSize buffers) {
  // FIXME: Use read(into, buffers) and then sort out in case of
  // failure, the readv probably succeed directly with much less
  // overhead.

  assert(!buffers || into);

  // Keep reading as long as possible.  Let system errors fly
  // over us, they are a hard error.
  IOSize x;
  IOSize done = 0;
  for (IOSize i = 0; i < buffers; ++i) {
    done += (x = xread(into[i]));
    if (x < into[i].size())
      break;
  }
  return done;
}

//////////////////////////////////////////////////////////////////////
IOSize Storage::write(IOBuffer from, IOOffset pos) { return write(from.data(), from.size(), pos); }

IOSize Storage::write(const void *from, IOSize n, IOOffset pos) {
  // FIXME: this is not thread safe!  split into separate interface
  // that a particular storage can choose to support or not?  make
  // sure that throw semantics are correct here!

  // FIXME: use saveposition object in case exceptions are thrown?
  IOOffset here = position();
  position(pos);
  n = write(from, n);
  position(here);
  return n;
}

IOSize Storage::writev(const IOPosBuffer *from, IOSize n) {
  IOSize total = 0;
  for (IOSize i = 0; i < n; ++i) {
    try {
      total += write(from[i].data(), from[i].size(), from[i].offset());
    } catch (cms::Exception &) {
      if (!total)
        throw;
      break;
    }
  }
  return total;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** @fn IOSize Storage::write (const void *from, IOSize n)
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
IOSize Storage::write(unsigned char byte) { return write(&byte, 1); }

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
IOSize Storage::write(IOBuffer from) { return write(from.data(), from.size()); }

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
IOSize Storage::writev(const IOBuffer *from, IOSize buffers) {
  assert(!buffers || from);

  // Keep writing as long as possible; ignore errors if we have
  // written something, otherwise pass it on.
  IOSize x;
  IOSize done = 0;
  try {
    for (IOSize i = 0; i < buffers; ++i) {
      done += (x = write(from[i].data(), from[i].size()));
      if (x < from[i].size())
        break;
    }
  } catch (cms::Exception &) {
    if (!done)
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
IOSize Storage::xwrite(IOBuffer from) { return xwrite(from.data(), from.size()); }

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
IOSize Storage::xwrite(const void *from, IOSize n) {
  // Keep writing until we've written it all.  Let errors fly over.
  IOSize done = 0;
  while (done < n)
    done += write((const char *)from + done, n - done);

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
IOSize Storage::xwritev(const IOBuffer *from, IOSize buffers) {
  // Keep writing until we've written it all.  Let errors fly over.
  // FIXME: Use writev(from, buffers) and then sort out in case of
  // failure, the writev probably succeed directly with much less
  // overhead.
  assert(!buffers || from);

  IOSize done = 0;
  for (IOSize i = 0; i < buffers; ++i)
    done += xwrite(from[i].data(), from[i].size());

  return done;
}

//////////////////////////////////////////////////////////////////////
IOOffset Storage::position() const {
  Storage *self = const_cast<Storage *>(this);
  return self->position(0, CURRENT);
}

IOOffset Storage::size() const {
  // FIXME: use saveposition object in case exceptions are thrown?
  Storage *self = const_cast<Storage *>(this);
  IOOffset here = position();
  self->position(0, END);
  IOOffset size = position();
  self->position(here);  // FIXME: VERIFY()?
  return size;
}

void Storage::rewind() { position(0); }

//////////////////////////////////////////////////////////////////////
bool Storage::prefetch(const IOPosBuffer * /* what */, IOSize /* n */) { return false; }

//////////////////////////////////////////////////////////////////////
void Storage::flush() {}

void Storage::close() {}

//////////////////////////////////////////////////////////////////////
bool Storage::eof() const { return position() == size(); }
