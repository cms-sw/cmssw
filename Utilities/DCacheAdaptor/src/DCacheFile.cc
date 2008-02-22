
// $Id$

#include "Utilities/DCacheAdaptor/interface/DCacheFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <unistd.h>
#include <fcntl.h>
#include <dcap.h>

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
    if (m_close) {
      edm::LogError("DCacheFileError")
        << "DCacheFile destructor called but file is still open";
    }
}

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
    m_name = name;

    // Actual open
    if ( (name == 0) || (*name == 0) ) {
      throw cms::Exception("DCacheFile")
        << "DCacheFile::open() called with a file name that is empty or NULL";
    }
    if ( (flags & (IOFlags::OpenRead | IOFlags::OpenWrite)) == 0) {
      throw cms::Exception("DCacheFile")
        << "DCacheFile::open() called with flag not set for read nor write";
    }

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
    if ((newfd = dc_open (name, openflags, perms.native ())) == -1) {
      throw cms::Exception("DCacheFile")
        << "  DCacheFile::open()"
        << "\n  dc_open failed: filename = " << m_name
        << "\n  open flags = " << openflags
        << "\n  permissions = " << perms.native()
        << "\n  dcache error code = " << dc_errno;
    }

    m_fd = newfd;

    // Disable buffering in dCache library?  This can make dramatic
    // difference to the system and client performance (factors of
    // ten difference in the amount of data read, and time spent
    // reading). Note also that docs say the flag turns off write
    // buffering -- this turns off all buffering.
    if (flags & IOFlags::OpenUnbuffered)
	dc_noBuffering (m_fd);

    m_close = true;

    edm::LogInfo("DCacheFileInfo")
      << "Opened file " << m_name;
}

void
DCacheFile::close (void)
{
    if (m_fd == IOFD_INVALID) {
      edm::LogError("DCacheFileError")
        << "DCacheFile::close() called but the file is not open";
      m_close = false;
      return;
    }

    dc_errno = 0;
    if (dc_close (m_fd) == -1) {
      edm::LogWarning("DCacheFileWarning")
        << "DCacheFile::close() - error in dc_close() " << dc_errno;
    }

    m_close = false;
    m_fd = IOFD_INVALID;

    edm::LogInfo("DCacheFileInfo")
      << "Closed file " << m_name;
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

IOSize
DCacheFile::read (void *into, IOSize n)
{
    int outputLocation = 0;
    int neededBytes = n;

    // We need a loop here because dc_read can return the 
    // number of bytes you requested or some number less than
    // what you requested.  This is not considered an error
    // state, but normal behavior.  (It was described to me
    // as POSIX behavior, although personally it sounds like
    // a bug in dc_read to me, maybe there is some technical
    // reason that this is sensible behavior).  Anyway, if
    // you get less bytes than you requested you call dc_read
    // again as many times as it takes to get all the bytes you
    // want.
    while (neededBytes > 0) {
 
      dc_errno = 0;
      ssize_t s = dc_read(m_fd, static_cast<unsigned char*>(into) + outputLocation, neededBytes);
      if (s == -1) {
        throw cms::Exception("DCacheFile")
          << "  dc_read() failed: filename = " << m_name
          << "\n  requested " << neededBytes << " bytes"
          << "\n  read " << s << " bytes (negative indicates error)"
          << "\n  dcache error code = " << dc_errno;
      }

      neededBytes -= s;
      outputLocation += s;

      // If you hit end of file, dc_read gets the number of bytes before the
      // end of file and returns that number of bytes.  It is not an error and
      // dc_errno is 0.  Subsequent calls to dc_read will return 0 and dc_errno
      // will be 0.  Again, not an error condition.  This I determined empirically.
      // I do not know how one determines the difference between end of file
      // and dc_read just returning 0.  So here I just assume if it returns 0
      // that means end of file.  I do not know if that is always correct or not.
      if (s == 0) break;
    }
    return (n - neededBytes);
}

IOSize
DCacheFile::write (const void *from, IOSize n)
{
    dc_errno = 0;
    ssize_t s = dc_write (m_fd, from, n);
    if (s == -1)
      throw cms::Exception("DCacheFile")
        << "  dc_write() failed: filename = " << m_name
        << "\n  requested " << n << " bytes"
        << "\n  wrote " << s << " bytes (negative indicates error)"
        << "\n  dcache error code = " << dc_errno;

    return s >= 0 ? s : 0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
DCacheFile::position (IOOffset offset, Relative whence /* = SET */)
{
    if (m_fd == IOFD_INVALID) {
        throw cms::Exception("DCacheFile")
          << "DCacheFile::position() called but the file is not open\n";
    }
    if (whence != CURRENT && whence != SET &&  whence != END) {
      throw cms::Exception("DCacheFile")
        << "DCacheFile::position() called with undefined position\n";
    }

    IOOffset	result;
    int		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);

    dc_errno = 0;
    if ((result = dc_lseek64 (m_fd, offset, mywhence)) == -1) {
      throw cms::Exception("DCacheFile")
        << "dc_lseek64() failed"
        << " with dcache error code = " << dc_errno;
    }
    // FixMe when they fix it....
    if (whence == SEEK_END)
      if ((result = dc_lseek64 (m_fd, result, SEEK_SET))== -1) {
        throw cms::Exception("DCacheFile")
          << "dc_lseek64() failed"
          << " with dcache error code = " << dc_errno;
      }
    
    return result;
}

void
DCacheFile::resize (IOOffset /* size */)
{
    throw cms::Exception("DCacheFile")
      << "DCacheFile::resize() called but this function is not implemented yet";
}
