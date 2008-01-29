//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/DCacheStorageMaker.h"
#include "Utilities/DCacheAdaptor/interface/DCacheFile.h"
#include<unistd.h>
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

/** Return appropriate path for use with dcap library.  If the path is in the
    URL form ([gsi]dcap://host:port/path), return the full URL as it is.  If
    the path is /pnfs form (dcap:/pnfs/path), return only the path part, unless
    the protocol is 'gsidcap', in which case return the whole thing.  */
std::string
DCacheStorageMaker::pathForUrl (const std::string &proto, const std::string &path)
{
  size_t p=path.find("/pnfs");
  if (p<3) return (proto == "gsidcap") ? proto + ':'+ path.substr(p) : path.substr(p);
  // remove multiple "/"
  p=path.find_first_not_of('/');
  // it's url, return the full thing
  return proto + "://" + path.substr(p);
}


/** Open a storage object for the given URL (protocol + path), using the
    @a mode bits.  No temporary files are downloaded.  */
seal::Storage *
DCacheStorageMaker::doOpen (const std::string &proto,
			  const std::string &path,
			  int mode,
			  const std::string & /* tmpdir */)
{ return new DCacheFile (pathForUrl (proto, path), mode); }

/** Check if the given URL (protocol + path) exists, and if so and @a size is
    non-null, update @a size to the file's size.  */
bool	  
DCacheStorageMaker::doCheck (const std::string &proto,
			   const std::string &path,
			   seal::IOOffset *size /* = 0 */)
{
    std::string testpath (pathForUrl (proto, path));
    if (dc_access (testpath.c_str (), R_OK) != 0)
      return false;

    if (size)
    {
	struct stat64 buf;
	if (dc_stat64 (testpath.c_str (), &buf) != 0)
	  return false;
	
	*size = buf.st_size;
    }
    
    return true;
}
