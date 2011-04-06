#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/src/SysFile.h"
#include "Utilities/StorageFactory/src/Throw.h"
#include <cassert>

using namespace IOFlags;

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

/** @fn IOSize File::read (IOBuffer into, IOOffset pos)
    Read from the file at the specified position.  */

/** @fn IOSize File::write (IOBuffer from, IOOffset pos)
    Write to the file at the specified position.  */

/** @fn IOOffset File::size (void) const
    Get the size of the file.  */

/** @fn IOOffset File::position (void) const
    Return the current file pointer position.  */

/** @fn IOOffset File::position (IOOffset offset, Relative whence = SET)
    Move the current file pointer to @a offset relative to @a whence.
    Returns the new file offset relative to the beginning of the file. */

/** @fn void  File::resize (IOOffset size)
    Resize to the file to @a size.  If @a size is less than the file's
    current size, the file is truncated.  If @a size is larger than
    the file's current size, the file is extended with zero bytes.
    Does not change the current file pointer.  */

/** @fn void File::flush (void)
    Flush the system's file system buffers for this file.  */

/** @fn bool File::sysclose (IOFD fd, int *error)
    Actually close a file handle and return error code.  */

/** @fn void File::times (Time *ctime, Time *mtime, Time *atime) const
    Get the file's creation, modification and access times.  Fills in
    non-null #Time parameters.  */

/** @fn bool File::status (IOStatus &s) const
    Get the full #IOStatus description of the file.  */

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
/** Create a new file object without a file attached to it.  */
File::File (void)
{
  fd (EDM_IOFD_INVALID);
  m_flags = 0;
}

/** Create a new file object from a file descriptor.  The descriptor
    will be closed automatically when the file object is destructed
    if @a autoclose is @c true (the default).  */
File::File (IOFD fd, bool autoclose /* = true */)
{
  this->fd (fd);
  m_flags = autoclose ? InternalAutoClose : 0;
}

/** Internal function for copying file objects to retain the state flags. */
File::File (IOFD fd, unsigned flags)
{
  this->fd (fd);
  m_flags = flags;
}

/** Create a new file object by calling #open() with the given arguments.  */
File::File (const char *name, int flags /*= OpenRead*/, int perms /*= 0666*/)
{ open (name, flags, perms); }

/** Create a new file object by calling #open() with the given arguments.  */
File::File (const std::string &name, int flags /*= OpenRead*/, int perms /*= 0666*/)
{ open (name.c_str (), flags, perms); }

/** Release the resources held by the file object.  If the object
    holds a valid file descriptor given to it through the constructor
    or obtained by calling #open(), the descriptor will be closed.  */
File::~File (void)
{
  if (m_flags & InternalAutoClose)
    abort ();
}

//////////////////////////////////////////////////////////////////////
/** Set the autoclose flag of the file.  If @a autoclose is @c true,
    the destructor will automatically try to close the underlying file
    descriptor.  Otherwise the file descriptor will be left open.  Set
    the flag off if the file descriptor is originally owned by someone
    else.  */
void
File::setAutoClose (bool autoclose)
{
  m_flags &= ~InternalAutoClose;
  if (autoclose)
    m_flags |= InternalAutoClose;
}

//////////////////////////////////////////////////////////////////////
/** Duplicate the file object.  If @a copy, also duplicates the
    underlying file descriptor, otherwise the two will point to the
    same descriptor.  If the file descriptor is not copied, the copy
    will not close its file descriptor on destruction, the original
    object (@c this) will. */
File *
File::duplicate (bool copy) const
{
  File *dup = new File (fd (), copy ? m_flags : 0);
  return copy ? this->duplicate (dup) : dup;
}

/** Internal implementation of #duplicate() to actually duplicate the
    file handle into @a child. */
File *
File::duplicate (File *child) const
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);
  assert (child);
  child->fd (sysduplicate (fd));
  child->m_flags = m_flags;
  return child;
}

//////////////////////////////////////////////////////////////////////
/** Create and open the file @a name in write mode.  If @a exclusive,
    the creation fails if the file already exists, otherwise if the
    file exists, it will be truncated.  The new file will have the
    permissions @a perms. */
void
File::create (const char *name, bool exclusive /*=false*/, int perms/*=0666*/)
{
  open (name,
	(OpenCreate | OpenWrite | OpenTruncate
	 | (exclusive ? OpenExclusive : 0)),
	perms);
}

/** Create and open the file @a name in write mode.  If @a exclusive,
    the creation fails if the file already exists, otherwise if the
    file exists, it will be truncated.  The new file will have the
    permissions @a perms. */
void
File::create (const std::string &name, bool exclusive /*=false*/, int perms/*=0666*/)
{
  open (name.c_str (),
	(OpenCreate | OpenWrite | OpenTruncate
	 | (exclusive ? OpenExclusive : 0)),
	perms);
}

/** Open or possibly create the file @a name with options specified in
    @a flags.  If the file is to be created, it will be given the
    permissions @a perms.  If this object already has a file open,
    it is closed first.  Redirected to the overloaded method taking
    a "const char *" argument.  */
void
File::open (const std::string &name, int flags /*= OpenRead*/, int perms /*= 0666*/)
{ open (name.c_str (), flags, perms); }

/** Open or possibly create the file @a name with options specified in
    @a flags.  If the file is to be created, it will be given the
    permissions @a perms.  If this object already has a file open,
    it is closed first.  */
void
File::open (const char *name, int flags /*= OpenRead*/, int perms /*= 0666*/)
{
  // is zero and always implied.  OTOH, existence check should be
  // done with Filename::exists() -- see comments there about what
  // can happen on a WIN32 remote share even if the file doesn't
  // exist.  For now make sure that read or write was asked for.

  assert (name && *name);
  assert (flags & (OpenRead | OpenWrite));

  // If I am already open, close the old file first.
  if (fd () != EDM_IOFD_INVALID && (m_flags & InternalAutoClose))
    close ();
    
  IOFD		newfd = EDM_IOFD_INVALID;
  unsigned	newflags = InternalAutoClose;

  sysopen (name, flags, perms, newfd, newflags);

  fd (newfd);
  m_flags = newflags;
}

void
File::attach (IOFD fd)
{
  this->fd (fd);
  m_flags = 0;
}

//////////////////////////////////////////////////////////////////////
/** Prefetch data for the file.  */
bool
File::prefetch (const IOPosBuffer *what, IOSize n)
{
  IOFD fd = this->fd();
  for (IOSize i = 0; i < n; ++i)
  {
#if F_RDADVISE
    radvisory info;
    info.ra_offset = what[i].offset();
    info.ra_count = what[i].size();
    fcntl(fd, F_RDADVISE, &info);
#elif _POSIX_ADVISORY_INFO > 0
    posix_fadvise(fd, what[i].offset(), what[i].size(), POSIX_FADV_WILLNEED);
#else
# error advisory read ahead not available on this platform
#endif
  }
  return true;
}

/** Read from the file.  */
IOSize
File::read (void *into, IOSize n)
{ return IOChannel::read (into, n); }

/** Read from the file.  */
IOSize
File::readv (IOBuffer *into, IOSize length)
{ return IOChannel::readv (into, length); }

/** Write to the file.  */
IOSize
File::write (const void *from, IOSize n)
{
  // FIXME: This may create a race condition or cause trouble on
  // remote files.  Should be currently needed only on WIN32.
  if (m_flags & OpenAppend)
    position (0, END);

  IOSize s = IOChannel::write (from, n);

  if (m_flags & OpenUnbuffered)
    // FIXME: Exception handling?
    flush ();

  return s;
}

/** Write to the file.  */
IOSize
File::writev (const IOBuffer *from, IOSize length)
{
  // FIXME: This may create a race condition or cause trouble on
  // remote files.  Should be currently needed only on WIN32.
  if (m_flags & OpenAppend)
    position (0, END);

  IOSize s = IOChannel::writev (from, length);

  if (m_flags & OpenUnbuffered)
    // FIXME: Exception handling?
    flush ();

  return s;
}

/** Close the file.  */
void
File::close (void)
{
  IOFD fd = this->fd ();
  assert (fd != EDM_IOFD_INVALID);

  int error;
  if (! sysclose (fd, &error))
    throwStorageError("FileCloseError", "Calling File::close()",
                      "sysclose", error);

  m_flags &= ~InternalAutoClose;
  this->fd (EDM_IOFD_INVALID);
}

/** Close the file and ignore all errors.  */
void
File::abort (void)
{
  IOFD fd = this->fd ();
  if (fd != EDM_IOFD_INVALID)
  {
    sysclose (fd);
    m_flags &= ~InternalAutoClose;
    this->fd (EDM_IOFD_INVALID);
  }
}
