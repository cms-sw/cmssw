#include "Utilities/LStoreAdaptor/interface/LStoreFile.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <dlfcn.h>
#include <iostream>
#include <string.h>

// dlsym isn't reentrant, need a locak around it
pthread_mutex_t LStoreFile::m_dlopen_lock = PTHREAD_MUTEX_INITIALIZER;

LStoreFile::LStoreFile (void)
  : m_fd (NULL),
    m_close (false),
	m_is_loaded(false)
{
	loadLibrary();	
}

LStoreFile::LStoreFile (void * fd)
  : m_fd (fd),
    m_close (true),
	m_is_loaded(false)
{
	loadLibrary();	
}

LStoreFile::LStoreFile (const char *name,
    	    int flags /* = IOFlags::OpenRead */,
    	    int perms /* = 066 */ )
  : m_fd (NULL),
    m_close (false),
	m_is_loaded(false)
{   loadLibrary();
	open (name, flags, perms); }

LStoreFile::LStoreFile (const std::string &name,
    	    int flags /* = IOFlags::OpenRead*/, 
    	    int perms /* = 066 */)
  : m_fd (NULL),
    m_close (false),
	m_is_loaded(false)
{   loadLibrary();
	open (name.c_str (), flags, perms); }

LStoreFile::~LStoreFile (void)
{
  if (m_close)
    edm::LogError("LStoreFileError")
      << "Destructor called on LStore file '" << m_name
      << "' but the file is still open";
  closeLibrary();
}


// Helper macro to perform dlsym()
// Double cast is supposed to be more compliant
// otherwise, GCC complains with the pointer conversion
#define REDD_LOAD_SYMBOL( NAME, TYPE ) 	dlerror();\
    NAME = reinterpret_cast<TYPE>(reinterpret_cast<size_t>( \
				dlsym(m_library_handle, #NAME))); \
	if ( (retval = dlerror()) ) {\
		throw cms::Exception("LStoreFile::loadLibrary()") <<\
			"Failed to load dlsym LStore library: " << retval;\
	}\
	if ( NAME == NULL) {\
		throw cms::Exception("LStoreFile::loadLibrary()") <<\
			"Got a null pointer back from dlsym()\n";\
	}

void LStoreFile::loadLibrary() {
	edm::LogError("LStoreFile::loadLibrary()") << "Loading library\n";
	LStoreFile::MutexWrapper lockObj( & this->m_dlopen_lock );
	// until ACCRE removes the java dependency from their client libs,
	// we'll dlopen() them so they don't need to be brought along with cmssw
	// if you're running LStore at your site, you will have the libs anyway
	// TODO add wrappers to make this work in OSX as well (CMSSW's getting ported?)
	// TODO   should be easy, just need to know the "proper" way to do #if OSX
	// -Melo

	m_library_handle =
	     dlopen("libreddnet.so", RTLD_LAZY);
	if (m_library_handle == NULL) {
		throw cms::Exception("LStoreFile::loadLibrary()")
			<< "Can't dlopen() LStore libraries: " << dlerror();
	}

	char * retval = NULL;
	// Explicitly state the size of these values, keeps weird 64/32 bit stuff away
	REDD_LOAD_SYMBOL( redd_init, int32_t(*)()); 
	REDD_LOAD_SYMBOL( redd_read, int64_t(*)(void *, char*, int64_t));
	REDD_LOAD_SYMBOL( redd_lseek, int64_t(*)(void*, int64_t, uint32_t));
	REDD_LOAD_SYMBOL( redd_open, void*(*)(const char*,int,int));
	REDD_LOAD_SYMBOL( redd_write, int64_t(*)(void *, const char *, int64_t));
	REDD_LOAD_SYMBOL( redd_term, int32_t(*)());
	REDD_LOAD_SYMBOL( redd_errno,  int32_t(*)());
	REDD_LOAD_SYMBOL( redd_strerror, const std::string & (*)());

	if ( (*redd_init)() ) {
		throw cms::Exception("LStoreFile::loadLibrary()")
			<< "Error in redd_init: " << (*redd_strerror)();
	}
	m_is_loaded = true;

}

void LStoreFile::closeLibrary() {
	try {
		LStoreFile::MutexWrapper lockObj( & this->m_dlopen_lock );
		
		// What is the correct idiom for propagating error messages 
		// in functions that are exclusively called in destructors?
		// Seriously. I have no idea
		// melo
		if ( m_is_loaded ) {
			if ( (*redd_term)() ) {
				throw cms::Exception("LStoreFile::closeLibrary()")
					<< "Error in redd_term: " << (*redd_strerror)();
			}
		}
		if ( m_library_handle != NULL ) {
			if ( dlclose( m_library_handle ) ) {
				throw cms::Exception("LStoreFile::closeLibrary()")
					<< "Error on dlclose(): " << dlerror();
			}
		}
	} catch (cms::Exception & e) {
		edm::LogError("LStoreFileError")
	      << "LStoreFile had an error in its destructor: " << e;
	}
	m_is_loaded = false;
}


//////////////////////////////////////////////////////////////////////
void
LStoreFile::create (const char *name,
		    bool exclusive /* = false */,
		    int perms /* = 066 */)
{
  open (name,
        (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
         | (exclusive ? IOFlags::OpenExclusive : 0)),
        perms);
}

void
LStoreFile::create (const std::string &name,
                    bool exclusive /* = false */,
                    int perms /* = 066 */)
{
  open (name.c_str (),
        (IOFlags::OpenCreate | IOFlags::OpenWrite | IOFlags::OpenTruncate
         | (exclusive ? IOFlags::OpenExclusive : 0)),
        perms);
}

void
LStoreFile::open (const std::string &name,
                  int flags /* = IOFlags::OpenRead */,
                  int perms /* = 066 */)
{ open (name.c_str (), flags, perms); }

void
LStoreFile::open (const char *name,
                  int flags /* = IOFlags::OpenRead */,
                  int perms /* = 066 */)
{
  m_name = name;

  // Actual open
  if ((name == 0) || (*name == 0))
    throw cms::Exception("LStoreFile::open()")
      << "Cannot open a file without a name";

  if ((flags & (IOFlags::OpenRead | IOFlags::OpenWrite)) == 0)
    throw cms::Exception("LStoreFile::open()")
      << "Must open file '" << name << "' at least for read or write";

  // If I am already open, close old file first
  if (m_fd != NULL && m_close)
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

  void * newfd = NULL;
  if ((newfd = (*redd_open) (name, openflags, perms)) == NULL)
    throw cms::Exception("LStoreFile::open()")
      << "redd_open(name='" << name
      << "', flags=0x" << std::hex << openflags
      << ", permissions=0" << std::oct << perms << std::dec
      << ") => error '" << (*redd_strerror)()
      << "' (redd_errno=" << (*redd_errno)() << ")";

  m_fd = newfd;

  m_close = true;

  edm::LogInfo("LStoreFileInfo") << "Opened " << m_name;
}

void
LStoreFile::close (void)
{
  if (m_fd == NULL)
  {
    edm::LogError("LStoreFileError")
      << "LStoreFile::close(name='" << m_name
      << "') called but the file is not open";
    m_close = false;
    return;
  }
  edm::LogInfo("LStoreFile::close()") << "closing " << m_name << std::endl;
  if ((*redd_close) (m_fd) == -1)
    edm::LogWarning("LStoreFileWarning")
      << "redd_close(name='" << m_name
      << "') failed with error '" << (*redd_strerror) ()
      << "' (redd_errno=" << (*redd_errno)() << ")";

  m_close = false;
  m_fd = NULL;

  // Caused hang.  Will be added back after problem is fixed.
  // edm::LogInfo("LStoreFileInfo") << "Closed " << m_name;
}

void
LStoreFile::abort (void)
{
  if (m_fd != NULL)
    (*redd_close) (m_fd);

  m_close = false;
  m_fd = NULL;
}


IOSize
LStoreFile::read (void *into, IOSize n)
{
  IOSize done = 0;
  while (done < n)
  {
    ssize_t s = (*redd_read) (m_fd, (char *) into + done, n - done);
    if (s == -1)
      throw cms::Exception("LStoreFile::read()")
        << "redd_read(name='" << m_name << "', n=" << (n-done)
        << ") failed with error '" << (*redd_strerror)()
        << "' (redd_errno=" << (*redd_errno)() << ")";
   done += s;
  }
  return done;
}

IOSize
LStoreFile::write (const void *from, IOSize n)
{
  IOSize done = 0;
  while (done < n)
  {
    ssize_t s = (*redd_write) (m_fd, (const char *) from + done, n - done);
    if (s == -1)
      throw cms::Exception("LStoreFile::write()")
        << "redd_write(name='" << m_name << "', n=" << (n-done)
        << ") failed with error '" << (*redd_strerror)()
        << "' (redd_errno=" << (*redd_errno)() << ")";
    done += s;
  }

  return done;
}
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
IOOffset
LStoreFile::position (IOOffset offset, Relative whence /* = SET */)
{
  if (m_fd == NULL)
    throw cms::Exception("LStoreFile::position()")
      << "LStoreFile::position() called on a closed file";
  if (whence != CURRENT && whence != SET && whence != END)
    throw cms::Exception("LStoreFile::position()")
      << "LStoreFile::position() called with incorrect 'whence' parameter";

  IOOffset	result;
  uint32_t		mywhence = (whence == SET ? SEEK_SET
		    	    : whence == CURRENT ? SEEK_CUR
			    : SEEK_END);
  if ((result = (*redd_lseek) (m_fd, (off_t) offset, (uint32_t)mywhence)) == -1)
    throw cms::Exception("LStoreFile::position()")
      << "redd_lseek64(name='" << m_name << "', offset=" << offset
      << ", whence=" << mywhence << ") failed with error '"
      << (*redd_strerror) () << "' (redd_errno=" << (*redd_errno)() << ")";
  return result;
}

void
LStoreFile::resize (IOOffset /* size */)
{
  throw cms::Exception("LStoreFile::resize()")
    << "LStoreFile::resize(name='" << m_name << "') not implemented";
}


////////////////////////////////////////////////////////////////////

LStoreFile::MutexWrapper::MutexWrapper( pthread_mutex_t * target ) 
{
	m_lock = target;
	pthread_mutex_lock( m_lock ); // never fails
}

LStoreFile::MutexWrapper::~MutexWrapper()
{
	int retval;
	if ( (retval = pthread_mutex_unlock( m_lock )) ) {
		// congrats. pthread_mutex_lock failed and we're in a destructor
		// I don't know what to do here
		// Then again, if the mutex is jammed, things are already boned
		// Cry for a second, then continue with life, I guess
    	// melo

		char buf[1024];
		edm::LogError("LStoreFileError")
	      << "LStoreFile couldn't unlock a mutex. Not good." 
		  << strerror_r( retval, buf, 1024 );
	}
}

