
// $Id: RFIOFile.cc,v 1.19 2007/09/12 21:25:14 wdd Exp $

#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIO.h"
#ifdef close
#undef close
#endif

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SealBase/TimeInfo.h"
#include <iostream>
#include <unistd.h>

RFIOFile::RFIOFile (void)
    : m_fd (IOFD_INVALID),
      m_close (false)
{}

RFIOFile::RFIOFile (IOFD fd)
    : m_fd (fd),
      m_close (true)
{}

RFIOFile::RFIOFile (const char *name,
		    int flags /* = IOFlags::OpenRead */,
		    FileAcl perms /* = 066 */)
    : m_fd (IOFD_INVALID),
      m_close (false)
{ open (name, flags, perms); }

RFIOFile::RFIOFile (const std::string &name,
		    int flags /* = IOFlags::OpenRead */,
		    FileAcl perms /* = 066 */)
    : m_fd (IOFD_INVALID),
      m_close (false)
{ open (name.c_str (), flags, perms); }

RFIOFile::~RFIOFile (void)
{
    if (m_close) {
      edm::LogError("RFIOFileError")
        << "RFIOFile destructor called but file is still open";
    }
}

//////////////////////////////////////////////////////////////////////

void
RFIOFile::create (const char *name,
		  bool exclusive /* = false */,
		  FileAcl perms /* = 066 */)
{
    open (name,
	  (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	   | (exclusive ? IOFlags::OpenExclusive : 0)),
	  perms);
}

void
RFIOFile::create (const std::string &name,
		  bool exclusive /* = false */,
		  FileAcl perms /* = 066 */)
{
    open (name.c_str (),
	  (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
	   | (exclusive ? IOFlags::OpenExclusive : 0)),
	  perms);
}

void
RFIOFile::open (const std::string &name,
		int flags /* = IOFlags::OpenRead */,
		FileAcl perms /* = 066 */)
{ open (name.c_str (), flags, perms); }

void
RFIOFile::open (const char *name,
		int flags /* = IOFlags::OpenRead */,
		FileAcl perms /* = 066 */)
{
  /// save history
  m_name = name;
  m_flags = flags;
  m_perms = perms;
  m_lastIOV.clear();

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

    if ( (name == 0) || (*name == 0) ) {
      throw cms::Exception("RFIOFile")
        << "RFIOFile::open() called with a file name that is empty or NULL";
    }
    if ( (flags & (IOFlags::OpenRead | IOFlags::OpenWrite)) == 0) {
      throw cms::Exception("RFIOFile")
        << "RFIOFile::open() called with flag not set for read nor write";
    }

    std::string lname (name);
    if (lname.find ("//")==0) lname.erase(0,1);

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
    if ((newfd = rfio_open64 (lname.c_str(), openflags, perms.native ())) == -1) {
      throw cms::Exception("RFIOFile")
        << "  RFIOFile::open()"
        << "\n  rfio_open failed: filename = " << lname
        << "\n  open flags = " << openflags
        << "\n  permissions = " << perms.native()
        << "\n  rfio error code = " << rfio_errno << " / " << serrno;
    }

    m_fd = newfd;
    m_close = true;
    m_currentPosition = 0;

    edm::LogInfo("RFIOFileInfo")
      << "Opened file " << lname;
}

void
RFIOFile::close (void)
{
    if (m_fd == IOFD_INVALID) {
      edm::LogError("RFIOFileError")
        << "RFIOFile::close() called but the file is not open";
      m_close = false;
      return;
    }

    serrno = 0;
    if (rfio_close64 (m_fd) == -1) {
      edm::LogWarning("RFIOFileWarning")
        << "RFIO::close(): error occurred in rfio_close64(), "
        << "rfio_errno = " << rfio_errno
        << "  serrno = " << serrno << std::endl;
      sleep(5);

      // When rfio_close64 fails then try the system close
      // function.  Added this on advice of Olof from CASTOR
      // operations.
      int status = ::close(m_fd);
      if (status < 0) {
        edm::LogWarning("RFIOFileWarning")
          << "RFIOFile::close(): Tried system level close after rfio_close64 failed\n"
          << "It also returns an error code: " << status;
      }
      else {
        edm::LogWarning("RFIOFileWarning")
          << "RFIOFile::close(): Tried system level close after rfio_close64 failed\n"
          << "It succeeded: " << status;
      }
    }

    m_close = false;
    m_fd = IOFD_INVALID;

    edm::LogInfo("RFIOFileInfo")
      << "Closed file " << m_name;
}

void
RFIOFile::abort (void)
{
  serrno = 0;
  if (m_fd != IOFD_INVALID)
	rfio_close64 (m_fd);

    m_close = false;
    m_fd = IOFD_INVALID;
}

/// close and open again...
void RFIOFile::reOpen() {

  // Remember the current position in the file
  IOOffset l_pos = m_currentPosition;

  close();
  sleep(5);
  open(m_name, m_flags, m_perms);

  // Set the position back to the same place it was
  // before the file close and open
  position(l_pos);
}

ssize_t
RFIOFile::retry_read (void *into, IOSize n, int max_retry /* =10 */) {

  // Retries are accomplished by this function recursively
  // calling itself.  Note that the first call in this series
  // of recursive function calls is not actually a retry, it
  // is the initial try.

  serrno=0;
  ssize_t s = rfio_read64 (m_fd, into, n);
  if ( (s == -1 && serrno == 1004) || (s>int(n)) ) {

    if (max_retry > 0) {

      // Wait a little while to give the CASTOR problem
      // causing the timeout a little time to clear.
      std::string sleepTime;
      int secondsToSleep = 5;
      if (max_retry >= 3) {
        sleepTime = "1 minute";
        secondsToSleep = 60;
      }
      else if (max_retry == 2) {
        sleepTime = "5 minutes";
        secondsToSleep = 300;
      }
      else if (max_retry == 1) {
        sleepTime = "10 minutes";
        secondsToSleep = 600;
      }

      edm::LogWarning("RFIOFileRetry")
        << "RFIOFile retrying read\n"
        << "  return value from rfio_read64 = " << s << "  (normally this is bytes read, -1 for error)\n"
        << "  bytes requested = " << n << "  (this and bytes read are equal unless error or EOF)\n"
        << "  serrno = " << serrno << "  (an error code rfio sets, 0 = OK, 1004 = timeout, ...)\n"
        << "  current position = " << m_currentPosition << "  (in bytes, beginning of file is 0)\n"
        << "  retries left before quitting = " << max_retry << "\n"
        << "  will close and reopen file " << m_name << "\n"
        << "  process waiting (sleeping) for " << sleepTime << " before attempting retry";
      edm::FlushMessageLog();

      sleep(secondsToSleep);

      // Improve the chances of success by closing and reopening
      // the file before retrying the read.  This also resets
      // the position in the file to the correct place.
      reOpen();

      max_retry--;
      return retry_read (into, n, max_retry);
    }
  }
  return s;
}

IOSize
RFIOFile::read (void *into, IOSize n)
{
  // Be aware that when enabled these LogDebug prints
  // will take more time than the read itself unless the reads
  // are proceeding slower than under optimal conditions.
  LogDebug("RFIOFileDebug")
    << "Entering RFIOFile read()";

  double start = seal::TimeInfo::realNsecs();

  serrno = 0;
  int maximumNumberOfRetries = 3;
  ssize_t s = retry_read (into, n, maximumNumberOfRetries);

  if (s == -1) {
    throw cms::Exception("RFIOFile")
      << "  rfio_read() failed: filename = " << m_name
      << "\n  requested " << n << " bytes"
      << "\n  read " << s << " bytes (negative indicates error)"
      << "\n  current position in file = " << m_currentPosition
      << "\n  rfio error code = " << rfio_errno << " / " << serrno;
  }

  m_currentPosition += s;

  double end = seal::TimeInfo::realNsecs();

  LogDebug("RFIOFileDebug")
    << "Exiting RFIOFile read()\n"
    << "  elapsed time = " << end - start << " ns\n"
    << "  bytes read = " << s << "\n"
    << "  file position = " << m_currentPosition;

  return s;
}

IOSize
RFIOFile::write (const void *from, IOSize n)
{
    serrno = 0;
    ssize_t s = rfio_write64 (m_fd, from, n);
    if (s == -1) {
      throw cms::Exception("RFIOFile")
        << "  rfio_write() failed: filename = " << m_name
        << "\n  requested " << n << " bytes"
        << "\n  wrote " << s << " bytes (negative indicates error)"
        << "\n  rfio error code = " << rfio_errno << " / " << serrno;
    }
    return s >= 0 ? s : 0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
RFIOFile::position (IOOffset offset, Relative whence /* = SET */)
{
    if (m_fd == IOFD_INVALID) {
      throw cms::Exception("RFIOFile")
        << "RFIOFile::position() called but the file is not open\n";
    }
    if (whence != CURRENT && whence != SET &&  whence != END) {
      throw cms::Exception("RFIOFile")
        << "RFIOFile::position() called with undefined position\n";
    }

    serrno = 0;
    IOOffset	result;
    int		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);

    if ((result = rfio_lseek64 (m_fd, offset, mywhence)) == -1) {
      throw cms::Exception("RFIOFile")
        << "rfio_lseek() failed"
        << " with rfio error code = " << rfio_errno << " / " << serrno;
    }

    m_currentPosition = result;

    return result;
}

void
RFIOFile::resize (IOOffset /* size */)
{
  throw cms::Exception("RFIOFile")
    << "RFIOFile::resize() called but this function is not implemented yet";
}

void          
RFIOFile::preseek(const IOVec& iov) {
  m_lastIOV = iov;

  if (rfioreadopt (RFIO_READOPT)!=1) { 
    throw cms::Exception("RFIOFile")
      << "RFIOFile::preseek() called with readopt != 1";
  }

  serrno = 0;
  int max_retry = 5;
  while ( rfio_preseek64(m_fd, 
			 const_cast<struct iovec64*>(&iov[0]), iov.size()) == -1) {
    if (max_retry == 0) { 
      throw cms::Exception("RFIOFile")
        << "RFIOFile::preseek; rfio_preseek() failed and retries exhausted"
        << "\nLast failure with rfio error code = " << rfio_errno << " / " << serrno;
    }

    edm::LogWarning("RFIOFileWarning")
      << "error in RFIO preseek: retry";
    max_retry--;
    sleep(5);
    serrno = 0;
  }
}

