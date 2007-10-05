//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/LocalStorageMaker.h"
#include "SealBase/IOStatus.h"
#include "SealBase/Filename.h"
#include "SealBase/File.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

seal::Storage *
LocalStorageMaker::doOpen (const std::string & /* proto */,
			 const std::string &path,
			 int mode,
			 const std::string & /* tmpdir */)
{ 
  // FIXME
  // Force unbuffered mode (bypassing page cache) off.  We
  // currently make so small reads that unbuffered access
  // will cause significant system load.  The unbuffered
  // hint is really for networked files (rfio, dcap, etc.),
  // where we don't want extra caching on client side due
  // non-sequential access patterns.
  mode &= ~seal::IOFlags::OpenUnbuffered;
  
  return new seal::File (path, mode); 
}

bool
LocalStorageMaker::doCheck (const std::string &proto,
		          const std::string &path,
		          seal::IOOffset *size /* = 0 */)
{
    seal::Filename name (path);
    if (! name.exists ())
	return false;

    if (size)
    {
	seal::IOStatus stat;
	name.status (stat);
	*size = stat.m_size;
    }

    return true;
}
