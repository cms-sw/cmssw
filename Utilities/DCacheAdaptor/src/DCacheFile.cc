//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/DCacheAdaptor/interface/DCacheFile.h"
#include "Utilities/DCacheAdaptor/interface/DCacheError.h"
#include "SealBase/DebugAids.h"
#include "SealBase/StringFormat.h"

#include <unistd.h>
#include <fcntl.h>
#include <dcap.h>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

DCacheFile::DCacheFile (void)
    : m_fd (IOFD_INVALID),
      m_close (false)
{}

DCacheFile::DCacheFile (IOFD fd)
    : m_fd (fd),
      m_close (true)
{}

DCacheFile::DCacheFile (const char *name,
		    int flags /* = IOFlags::OpenRead */,
		    FileAcl perms /* = 066 */)
    : m_fd (IOFD_INVALID),
      m_close (false)
{ open (name, flags, perms); }

DCacheFile::DCacheFile (const std::string &name,
		    int flags /* = IOFlags::OpenRead */,
		    FileAcl perms /* = 066 */)
    : m_fd (IOFD_INVALID),
      m_close (false)
{ open (name.c_str (), flags, perms); }

DCacheFile::~DCacheFile (void)
{
    if (m_close)
	abort ();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
void
DCacheFile::create (const char *name,
		  bool exclusive /* = false */,
		  FileAcl perms /* = 066 */)
{
    open (name,
	  (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	   | (exclusive ? IOFlags::OpenExclusive : 0)),
	  perms);
}

void
DCacheFile::create (const std::string &name,
		  bool exclusive /* = false */,
		  FileAcl perms /* = 066 */)
{
    open (name.c_str (),
	  (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	   | (exclusive ? IOFlags::OpenExclusive : 0)),
	  perms);
}

void
DCacheFile::open (const std::string &name,
		int flags /* = IOFlags::OpenRead */,
		FileAcl perms /* = 066 */)
{ open (name.c_str (), flags, perms); }

void
DCacheFile::open (const char *name,
		int flags /* = IOFlags::OpenRead */,
		FileAcl perms /* = 066 */)
{
    // Actual open
    ASSERT (name && *name);
    ASSERT (flags & (IOFlags::OpenRead | IOFlags::OpenWrite));

    // If I am already open, close old file first
    if (m_fd != IOFD_INVALID && m_close)
	close ();

    // Translate our flags to system flags
    int openflags = 0;

    if ((flags & IOFlags::OpenRead) && (flags & IOFlags::OpenWrite))
	openflags |= O_RDWR;
    else if (flags & IOFlags::OpenRead)
	openflags |= O_RDONLY;
    else if (flags & IOFlags::OpenWrite)
	openflags |= O_WRONLY;

    if (flags & IOFlags::OpenNonBlock)
	openflags |= O_NONBLOCK;

    if (flags & IOFlags::OpenAppend)
	openflags |= O_APPEND;

    if (flags & IOFlags::OpenCreate)
	openflags |= O_CREAT;

    if (flags & IOFlags::OpenExclusive)
	openflags |= O_EXCL;

    if (flags & IOFlags::OpenTruncate)
	openflags |= O_TRUNC;

    IOFD newfd = IOFD_INVALID;
    dc_errno = 0;
    if ((newfd = dc_open (name, openflags, perms.native ())) == -1)
	throw DCacheError (seal::StringFormat ("dc_open(%1,%2,%3").arg(name).arg(openflags).arg(perms.native()), dc_errno);

    m_fd = newfd;

    // Disable buffering in dCache library?  This can make dramatic
    // difference to the system and client performance (factors of
    // ten difference in the amount of data read, and time spent
    // reading). Note also that docs say the flag turns off write
    // buffering -- this turns off all buffering.
    if (flags & IOFlags::OpenUnbuffered)
	dc_noBuffering (m_fd);

    m_close = true;
}

void
DCacheFile::close (void)
{
    ASSERT (m_fd != IOFD_INVALID);

    dc_errno = 0;
    if (dc_close (m_fd) == -1)
	throw DCacheError ("dc_close()", dc_errno);

    m_close = false;
    m_fd = IOFD_INVALID;
}

void
DCacheFile::abort (void)
{
    if (m_fd != IOFD_INVALID)
	dc_close (m_fd);

    m_close = false;
    m_fd = IOFD_INVALID;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOSize
DCacheFile::read (void *into, IOSize n)
{
    dc_errno = 0;
    ssize_t s = dc_read (m_fd, into, n);
    if (s == -1)
	throw DCacheError ("dc_read()", dc_errno);

    return s;
}

IOSize
DCacheFile::write (const void *from, IOSize n)
{
    dc_errno = 0;
    ssize_t s = dc_write (m_fd, from, n);
    if (s == -1)
	throw DCacheError ("dc_write()", dc_errno);

    return s >= 0 ? s : 0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
DCacheFile::position (IOOffset offset, Relative whence /* = SET */)
{
    ASSERT (m_fd != IOFD_INVALID);
    ASSERT (whence == CURRENT || whence == SET || whence == END);

    IOOffset	result;
    int		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);

    dc_errno = 0;
    if ((result = dc_lseek64 (m_fd, offset, mywhence)) == -1)
	throw DCacheError ("dc_lseek()", dc_errno);
    // FixMe when they fix it....
    if (whence == SEEK_END)
      if ((result = dc_lseek64 (m_fd, result, SEEK_SET))== -1)
	throw DCacheError ("dc_lseek()", dc_errno);
    
    return result;
}

void
DCacheFile::resize (IOOffset /* size */)
{ throw DCacheError ("dc_ftruncate()", 0); }
