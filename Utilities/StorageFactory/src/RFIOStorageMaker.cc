//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/RFIOStorageMaker.h"
#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/RFIOAdaptor/interface/RFIO.h"

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
RFIOStorageMaker::open (const std::string & /* proto */,
		        const std::string &path,
			int mode,
		        const std::string & /* tmpdir */)
{ return new RFIOFile (path, mode); }

bool
RFIOStorageMaker::check (const std::string &proto,
		         const std::string &path,
		         seal::IOOffset *size /* = 0 */)
{
    if (rfio_access (path.c_str (), R_OK) != 0)
	return false;

    if (size)
    {
	struct stat buf;
	if (rfio_stat (path.c_str (), &buf) != 0)
	    return false;

	*size = buf.st_size;
    }

    return true;
}
