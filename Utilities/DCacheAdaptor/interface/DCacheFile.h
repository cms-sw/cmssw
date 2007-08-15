#ifndef DCACHE_ADAPTOR_DCACHE_FILE_H
# define DCACHE_ADAPTOR_DCACHE_FILE_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/Storage.h"
# include "SealBase/FileAcl.h"
# include "SealBase/IOFlags.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

using namespace seal;
/** DCache #Storage object.  */
class DCacheFile : public seal::Storage
{
public:
    DCacheFile (void);
    DCacheFile (IOFD fd);
    DCacheFile (const char *name, int flags = IOFlags::OpenRead, FileAcl perms = 0666);
    DCacheFile (const std::string &name, int flags = IOFlags::OpenRead, FileAcl perms = 0666);
    ~DCacheFile (void);

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

private:
    IOFD		m_fd;
    bool		m_close;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // DCACHE_ADAPTOR_DCACHE_FILE_H
