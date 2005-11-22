//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "SealBase/Filename.h"
#include "SealBase/TempFile.h"
#include "SealBase/Storage.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

StorageMaker::~StorageMaker (void)
{}

bool
StorageMaker::check (const std::string &proto,
		     const std::string &path,
		     seal::IOOffset *size /* = 0 */)
{
    using namespace seal;
    Filename tmpdir;
    TempFile::dir (tmpdir);

    bool found = false;
    int mode = IOFlags::OpenRead | IOFlags::OpenUnbuffered;
    if (Storage *s = open (proto, path, mode, tmpdir.name ()))
    {
	if (size)
	    *size = s->size ();

	s->close ();
	delete s;

	found = true;
    }

    Filename::remove (tmpdir, true);
    return found;
}
