//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIOError.h"
#include "Utilities/RFIOAdaptor/interface/RFIO.h"
#include "SealBase/DebugAids.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

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
{ m_close = false; open (name, flags, perms); }

RFIOFile::RFIOFile (const std::string &name,
		    int flags /* = IOFlags::OpenRead */,
		    FileAcl perms /* = 066 */)
    : m_fd (IOFD_INVALID),
      m_close (false)
{ open (name.c_str (), flags, perms); }

RFIOFile::~RFIOFile (void)
{
    if (m_close)
	abort ();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
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
    
    // Actual open
    ASSERT (name && *name);
    ASSERT (flags & (IOFlags::OpenRead | IOFlags::OpenWrite));

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
    if ((newfd = rfio_open64 (lname.c_str(), openflags, perms.native ())) == -1)
	throw RFIOError ("rfio_open()", rfio_errno, serrno);

    m_fd = newfd;
    m_close = true;
}

void
RFIOFile::close (void)
{
    ASSERT (m_fd != IOFD_INVALID);
    serrno = 0;

    if (rfio_close64 (m_fd) == -1)
	throw RFIOError ("rfio_close()", rfio_errno, serrno);

    m_close = false;
    m_fd = IOFD_INVALID;
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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

#include <iostream>

namespace {

  struct CountAndPrint {
    CountAndPrint(const std::string & message ) : m_n(0), m_message(message) {}
    ~CountAndPrint() {
      std::cerr << m_message << ": " << m_n << std::endl;
    }  
    int m_n;
    std::string m_message;
  };

  void countAndPrint() {
    static CountAndPrint c("total rfio retries");
    c.m_n++;
  }

}


/// close and open again...
void RFIOFile::reOpen() {
  std::cerr << "reopen " << m_name
	    << " with " << m_lastIOV.size() << " preseeks" << std::endl;
  close();
  sleep(5);
  open(m_name, m_flags, m_perms);
  if (!m_lastIOV.empty()) preseek(m_lastIOV);
}

ssize_t
RFIOFile::retry_read (void *into, IOSize n, int max_retry /* =10 */) {
  if (max_retry == 0) return -1;
  serrno=0;
  ssize_t s = rfio_read64 (m_fd, into, n);
  if ( (s == -1 && serrno == 1004) || (s>int(n)) 
       ) {
    countAndPrint();
    if (serrno == 1004) std::cerr << "timeout on RFIO read: retry" << std::endl;
    std::cerr << "current position " << m_currentPosition << std::endl;
    if (s>int(n)) std::cerr << "error in read " << n << ", " << s << std::endl;
    sleep(5);
    max_retry--;
    // close&open
    reOpen();
    // re seek
    IOOffset l_pos = m_currentPosition;
    // position(0); // hope reset internal rfio state
    position(l_pos);
    return retry_read (into, n, max_retry);
  } 
  return s;
}

IOSize
RFIOFile::read (void *into, IOSize n)
{
  serrno = 0;
  ssize_t s = retry_read (into, n);
  //    ssize_t s = rfio_read64 (m_fd, into, n);
  if (s == -1)
    throw RFIOError ("rfio_read()", rfio_errno, serrno);
  
  return s;
}

IOSize
RFIOFile::write (const void *from, IOSize n)
{
  serrno = 0;
    ssize_t s = rfio_write64 (m_fd, from, n);
    if (s == -1)
	throw RFIOError ("rfio_write()", rfio_errno, serrno);

    return s >= 0 ? s : 0;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
RFIOFile::position (IOOffset offset, Relative whence /* = SET */)
{
    ASSERT (m_fd != IOFD_INVALID);
    ASSERT (whence == CURRENT || whence == SET || whence == END);

    serrno = 0;
    IOOffset	result;
    int		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);

    if ((result = rfio_lseek64 (m_fd, offset, mywhence)) == -1)
	throw RFIOError ("rfio_lseek()", rfio_errno, serrno);

    m_currentPosition = result;
    return result;
}

void
RFIOFile::resize (IOOffset /* size */)
{ throw RFIOError ("rfio_ftruncate()", 0); }


void          
RFIOFile::preseek(const IOVec& iov) {
  m_lastIOV = iov;

  if (rfioreadopt (RFIO_READOPT)!=1) 
    throw RFIOError ("rfio_preseek(): readopt!=1", 0,0);


  serrno = 0;
  int max_retry = 5;
  while ( rfio_preseek64(m_fd, 
			 const_cast<struct iovec64*>(&iov[0]), iov.size()) == -1) {
    if (max_retry == 0)  
      throw RFIOError ("rfio_preseek()", rfio_errno, serrno);

    std::cerr << "error in RFIO preseek: retry" << std::endl;
    max_retry--;
    sleep(5);
    serrno = 0;
  }
}
