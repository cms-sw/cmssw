//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/GsiFTPStorageMaker.h"
#include "SealBase/File.h"
#include "SealBase/Filename.h"
#include "SealBase/TempFile.h"
#include "SealBase/SubProcess.h"

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
GsiFTPStorageMaker::doOpen (const std::string &proto,
		          const std::string &path,
			  int mode,
		          const std::string &tmpdir)
{
    using namespace seal;

    // FIXME: On write, create a temporary local file open for write;
    // on close, trigger transfer to destination.  If opening existing
    // file for write, may need to first download.
    ASSERT (! (mode & (IOFlags::OpenWrite | IOFlags::OpenCreate)));

    Filename	tmpname (Filename (tmpdir).asDirectory ());
    File	*tmpfile = TempFile::file (tmpname);
    std::string lurl = std::string ("file://") + tmpname.name ();
    std::string newurl ((proto == "sfn" ? "gsiftp" : proto) + ":" + path);
    const char	*ftpopts [] = { "globus-url-copy", newurl.c_str (), lurl.c_str (), 0 };
    SubProcess	ftp (ftpopts);
    int		rc = ftp.wait ();

    if (rc == 0)
    {
	// FIXME: How does caller know what it is called to delete it?
	tmpfile->rewind ();
	return tmpfile;
    }
    else
    {
	delete tmpfile;
	return 0;
    }
}
