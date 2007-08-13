#ifndef RFIO_ADAPTOR_RFIO_FILE_H
# define RFIO_ADAPTOR_RFIO_FILE_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/Storage.h"
# include "SealBase/FileAcl.h"
# include "SealBase/IOChannel.h"
# include "SealBase/IOFlags.h"

#include<vector>
#include <sys/stat.h>
// rfio data structure

#ifndef  RFIO_iovec64_H
#define  RFIO_iovec64_H
extern "C" {
struct iovec64 {
  off64_t iov_base;
  int iov_len ;
};
}
#endif

typedef std::vector<iovec64> IOVec;
inline void push(IOVec& vec, off64_t b, int l) {
  vec.push_back(iovec64());
  vec.back().iov_base = b;
  vec.back().iov_len = l;
} 

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

using namespace seal;
/** RFIO #Storage object.  */
class RFIOFile : public seal::Storage
{
public:
    RFIOFile (void);
    RFIOFile (IOFD fd);
    RFIOFile (const char *name, int flags = IOFlags::OpenRead, FileAcl perms = 0666);
    RFIOFile (const std::string &name, int flags = IOFlags::OpenRead, FileAcl perms = 0666);
    ~RFIOFile (void);

    virtual void	create (const char *name,
				bool exclusive = false,
				FileAcl perms = 0666);
    virtual void	create (const std::string &name,
				bool exclusive = false,
				FileAcl perms = 0666);
    virtual void	open (const char *name,
			      int flags = IOFlags::OpenRead,
			      FileAcl perms = 0666);
    virtual void	open (const std::string &name,
			      int flags = IOFlags::OpenRead,
			      FileAcl perms = 0666);

    using Storage::read;
    using Storage::write;
    using Storage::position;

    virtual IOSize	read (void *into, IOSize n);
    virtual IOSize	write (const void *from, IOSize n);

    virtual IOOffset	position (IOOffset offset, Relative whence = SET);
    virtual void	resize (IOOffset size);

    virtual void	close (void);
    virtual void	abort (void);

  virtual void          preseek(const IOVec& iov);

private:

    ssize_t     retry_read (void *into, IOSize n, int max_retry=10);

  void reOpen();

private:
    IOFD		m_fd;
    bool		m_close;

  /// history fro retry...
  std::string m_name;
  int m_flags;
  FileAcl m_perms;
  IOVec m_lastIOV;
  

  IOOffset              m_currentPosition;

  int m_nRetries;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // RFIO_ADAPTOR_RFIO_FILE_H
