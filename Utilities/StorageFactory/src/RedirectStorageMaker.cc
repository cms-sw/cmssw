//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/RedirectStorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

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
RedirectStorageMaker::open (const std::string &proto,
		            const std::string &path,
			    int mode,
		            const std::string &tmpdir)
{
    // Strip off proto and send rest back to factory.
    return StorageFactory::get ()->open (path, mode, tmpdir);
}
