#define __STDC_LIMIT_MACROS 1
#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIO.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cerrno>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <cstring>
#include <vector>

static double realNanoSecs (void)
{
  struct timespec ts;
  if (clock_gettime (CLOCK_REALTIME, &ts) == 0)
    return ts.tv_sec * 1e9 + ts.tv_nsec;
  return 0;
}

RFIOFile::RFIOFile (void)
  : m_fd (EDM_IOFD_INVALID),
    m_close (false),
    m_flags (0),
    m_perms (0),
    m_curpos (0)
{}

RFIOFile::RFIOFile (IOFD fd)
  : m_fd (fd),
    m_close (true),
    m_flags (0),
    m_perms (0),
    m_curpos (0)
{}

RFIOFile::RFIOFile (const char *name,
		    int flags /* = IOFlags::OpenRead */,
		    int perms /* = 066 */)
  : m_fd (EDM_IOFD_INVALID),
    m_close (false),
    m_flags (0),
    m_perms (0),
    m_curpos (0)
{ open (name, flags, perms); }

RFIOFile::RFIOFile (const std::string &name,
		    int flags /* = IOFlags::OpenRead */,
		    int perms /* = 066 */)
  : m_fd (EDM_IOFD_INVALID),
    m_close (false),
    m_flags (0),
    m_perms (0),
    m_curpos (0)
{ open (name.c_str (), flags, perms); }

RFIOFile::~RFIOFile (void)
{
  if (m_close)
    edm::LogError("RFIOFileError")
      << "Destructor called on RFIO file '" << m_name
      << "' but the file is still open";
}

//////////////////////////////////////////////////////////////////////

void
RFIOFile::create (const char *name,
		  bool exclusive /* = false */,
		  int perms /* = 066 */)
{
  open (name,
	(IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	 | (exclusive ? IOFlags::OpenExclusive : 0)),
	perms);
}

void
RFIOFile::create (const std::string &name,
		  bool exclusive /* = false */,
		  int perms /* = 066 */)
{
  open (name.c_str (),
	(IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	 | (exclusive ? IOFlags::OpenExclusive : 0)),
	perms);
}

void
RFIOFile::open (const std::string &name,
		int flags /* = IOFlags::OpenRead */,
		int perms /* = 066 */)
{ open (name.c_str (), flags, perms); }

void
RFIOFile::open (const char *name,
		int flags /* = IOFlags::OpenRead */,
		int perms /* = 066 */)
{
  // Save parameters for error recovery.
  m_name = name;
  m_flags = flags;
  m_perms = perms;

  // Reset RFIO error code.
  serrno = 0;

  // Disable buffering in rfio library?  Note that doing this on
  // one file disables it for everything.  Not much we can do...
  // but it does make a significant performance difference to the
  // clients.  Note also that docs say the flag turns off write
  // buffering -- this turns off all buffering.
  if (flags & IOFlags::OpenUnbuffered)
  {
    int readopt = 0;
    rfiosetopt (RFIO_READOPT, &readopt, sizeof (readopt));
  }
  else 
  {
    int readopt = 1;
    rfiosetopt (RFIO_READOPT, &readopt, sizeof (readopt));
  }

  if ((name == 0) || (*name == 0))
    throw cms::Exception("RFIOFile::open()")
      << "Cannot open a file without a name";

  if ((flags & (IOFlags::OpenRead | IOFlags::OpenWrite)) == 0)
    throw cms::Exception("RFIOFile::open()")
      << "Must open file '" << name << "' at least for read or write";

  std::string lname (name);
  if (lname.find ("//") == 0)
    lname.erase(0, 1);

  // If I am already open, close old file first
  if (m_fd != EDM_IOFD_INVALID && m_close)
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

  IOFD newfd = EDM_IOFD_INVALID;
  if ((newfd = rfio_open64 (lname.c_str(), openflags, perms)) == -1)
    throw cms::Exception("RFIOFile::open()")
      << "rfio_open(name='" << lname
      << "', flags=0x" << std::hex << openflags
      << ", permissions=0" << std::oct << perms << std::dec
      << ") => error '" << rfio_serror ()
      << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno << ")";

  m_fd = newfd;
  m_close = true;
  m_curpos = 0;

  edm::LogInfo("RFIOFileInfo") << "Opened " << lname;
}

void
RFIOFile::close (void)
{
  if (m_fd == EDM_IOFD_INVALID)
  {
    edm::LogError("RFIOFileError")
      << "RFIOFile::close(name='" << m_name
      << "') called but the file is not open";
    m_close = false;
    return;
  }

  serrno = 0;
  if (rfio_close64 (m_fd) == -1)
  {
    // If we fail to close the file, report a warning.
    edm::LogWarning("RFIOFileWarning")
      << "rfio_close64(name='" << m_name
      << "') failed with error '" << rfio_serror()
      << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno << ")";

    // When rfio_close64 fails then try the system close function as
    // per the advice from Olof Barring from the Castor operations.
    int status = ::close(m_fd);
    if (status < 0)
      edm::LogWarning("RFIOFileWarning")
        << "RFIOFile::close(): system level close after a failed"
        << " rfio_close64 also failed with error '" << strerror (errno)
        << "' (error code " << errno << ")";
    else
      edm::LogWarning("RFIOFileWarning")
        << "RFIOFile::close(): system level close after a failed"
        << " rfio_close64 succeeded";

    sleep(5);
  }

  m_close = false;
  m_fd = EDM_IOFD_INVALID;

  // Caused hang.  Will be added back after problem is fix
  // edm::LogInfo("RFIOFileInfo") << "Closed " << m_name;
}

void
RFIOFile::abort (void)
{
  serrno = 0;
  if (m_fd != EDM_IOFD_INVALID)
    rfio_close64 (m_fd);

  m_close = false;
  m_fd = EDM_IOFD_INVALID;
}

void RFIOFile::reopen (void)
{
  // Remember the current position in the file
  IOOffset lastpos = m_curpos;
  close();
  sleep(5);
  open(m_name, m_flags, m_perms);

  // Set the position back to the same place it was
  // before the file closed and opened.
  position(lastpos);
}

ssize_t
RFIOFile::retryRead (void *into, IOSize n, int maxRetry /* = 10 */)
{
  // Attempt to read up to maxRetry times.
  ssize_t s;
  do
  {
    serrno = 0;
    s = rfio_read64 (m_fd, into, n);
    if ((s == -1 && serrno == 1004) || (s > ssize_t (n)))
    {
      // Wait a little while to allow Castor to recover from the timeout.
      const char *sleepTimeMsg;
      int secondsToSleep = 5;
      switch (maxRetry)
      {
      case 1:
        sleepTimeMsg = "10 minutes";
        secondsToSleep = 600;
        break;

      case 2:
        sleepTimeMsg = "5 minutes";
        secondsToSleep = 300;
        break;

      default:
        sleepTimeMsg = "1 minute";
        secondsToSleep = 60;
      }

      edm::LogWarning("RFIOFileRetry")
        << "RFIOFile retrying read\n"
        << "  return value from rfio_read64 = " << s << " (normally this is bytes read, -1 for error)\n"
        << "  bytes requested = " << n << "  (this and bytes read are equal unless error or EOF)\n"
        << "  rfio error message = " << rfio_serror() << " (explanation from server, if possible)\n"
        << "  serrno = " << serrno << " (rfio server error code, 0 = OK, 1004 = timeout, ...)\n"
        << "  rfio_errno = " << rfio_errno << " (rfio error from actually accessing the file)\n"
        << "  current position = " << m_curpos << " (in bytes, beginning of file is 0)\n"
        << "  retries left before quitting = " << maxRetry << "\n"
        << "  will close and reopen file " << m_name << "\n"
        << "  will sleep for " << sleepTimeMsg << " before attempting retry";
      edm::FlushMessageLog();
      sleep(secondsToSleep);

      // Improve the chances of success by closing and reopening
      // the file before retrying the read.  This also resets
      // the position in the file to the correct place.
      reopen();
    }
    else
      break;
  } while (--maxRetry > 0);

  return s;
}

IOSize
RFIOFile::read (void *into, IOSize n)
{
  // Be aware that when enabled these LogDebug prints
  // will take more time than the read itself unless the reads
  // are proceeding slower than under optimal conditions.
  LogDebug("RFIOFileDebug") << "Entering RFIOFile read()";
  double start = realNanoSecs();

  ssize_t s;
  serrno = 0;
  if ((s = retryRead (into, n, 3)) < 0)
    throw cms::Exception("RFIOFile::read()")
      << "rfio_read(name='" << m_name << "', n=" << n << ") failed"
      << " at position " << m_curpos << " with error '" << rfio_serror()
      << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno << ")";

  m_curpos += s;

  double end = realNanoSecs();
  LogDebug("RFIOFileDebug")
    << "Exiting RFIOFile read(), elapsed time = " << end - start
    << " ns, bytes read = " << s << ", file position = " << m_curpos;

  return s;
}

IOSize
RFIOFile::readv (IOPosBuffer *into, IOSize buffers)
{
  if (! (m_flags & IOFlags::OpenUnbuffered))
    prefetch(into, buffers);
  return Storage::readv(into, buffers);
}

IOSize
RFIOFile::write (const void *from, IOSize n)
{
  serrno = 0;
  ssize_t s = rfio_write64 (m_fd, from, n);
  if (s < 0)
    throw cms::Exception("RFIOFile::write()")
      << "rfio_write(name='" << m_name << "', n=" << n << ") failed"
      << " at position " << m_curpos << " with error '" << rfio_serror()
      << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno << ")";
  return s;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
RFIOFile::position (IOOffset offset, Relative whence /* = SET */)
{
  if (m_fd == EDM_IOFD_INVALID)
    throw cms::Exception("RFIOFile::position()")
      << "RFIOFile::position() called on a closed file";
  if (whence != CURRENT && whence != SET && whence != END)
    throw cms::Exception("RFIOFile::position()")
      << "RFIOFile::position() called with incorrect 'whence' parameter";

  IOOffset	result;
  int		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);

  serrno = 0;
  if ((result = rfio_lseek64 (m_fd, offset, mywhence)) == -1)
    throw cms::Exception("RFIOFile::position()")
      << "rfio_lseek(name='" << m_name << "', offset=" << offset
      << ", whence=" << mywhence << ") failed at position "
      << m_curpos << " with error '" << rfio_serror()
      << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno << ")";

  m_curpos = result;
  return result;
}

void
RFIOFile::resize (IOOffset /* size */)
{
  throw cms::Exception("RFIOFile::resize()")
    << "RFIOFile::resize(name='" << m_name << "') not implemented";
}

bool
RFIOFile::prefetch (const IOPosBuffer *what, IOSize n)
{
  if (rfioreadopt (RFIO_READOPT) != 1)
    throw cms::Exception("RFIOFile::preseek()")
      << "RFIOFile::prefetch() called but RFIO_READOPT="
      << rfioreadopt (RFIO_READOPT) << " (must be 1)";

  std::vector<iovec64> iov (n);
  for (IOSize i = 0; i < n; ++i)
  {
    iov[i].iov_base = what[i].offset();
    iov[i].iov_len = what[i].size();
  }

  serrno = 0;
  int retry = 5;
  int result;
  while ((result = rfio_preseek64(m_fd, &iov[0], n)) == -1)
  {
    if (--retry <= 0)
    {
      edm::LogError("RFIOFile::prefetch")
        << "RFIOFile::prefetch(name='" << m_name << "') failed with error '"
	<< rfio_serror() << "' (rfio_errno=" << rfio_errno
	<< ", serrno=" << serrno << ")";
      return false;
    }
    else
    {
      edm::LogWarning("RFIOFileRetry")
        << "RFIOFile::prefetch(name='" << m_name << "') failed at position "
	<< m_curpos << " with error '" << rfio_serror()
        << "' (rfio_errno=" << rfio_errno << ", serrno=" << serrno
        << "); retrying " << (retry+1) << " times";
      serrno = 0;
      sleep(5);
    }
  }

  return true;
}
