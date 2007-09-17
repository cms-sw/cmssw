//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/HttpStorageMaker.h"
#include "SealBase/File.h"
#include "SealBase/Filename.h"
#include "SealBase/TempFile.h"
#include "SealBase/SubProcess.h"
#include "SealBase/PipeCmd.h"
#include "SealBase/StringOps.h"
#include <cstdlib>

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
HttpStorageMaker::doOpen (const std::string &proto,
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
    std::string newurl ((proto == "web" ? "http" : proto) + ":" + path);
    const char	*curlopts [] = {
	"curl", "-L", "-f", "-o", tmpname, "-q", "-s", "--url",
	newurl.c_str (), 0
    };
    SubProcess	curl (curlopts);
    int		rc = curl.wait ();

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

static std::string
readAllInput (const char **command)
{
    using namespace seal;

    PipeCmd	cmd (command, IOFlags::OpenRead);
    std::string	result;
    char	buf [1024];
    IOSize	n;

    while ((n = cmd.read (buf, sizeof (buf))))
	result.append (buf, n);

    cmd.wait ();
    return result;
}

bool
HttpStorageMaker::doCheck (const std::string &proto,
		         const std::string &path,
		         seal::IOOffset *size /* = 0 */)
{
    using namespace seal;

    // Try checking file existence on the server by fetching only the
    // headers with curl.  If this works, and the output includes the
    // "Content-Length" header, return that as the size.  Otherwise
    // fall back on the base class implementation: try open and return
    // info from that.  This improves performance when server can and
    // does answer us, but makes things slightly worse if it fails --
    // and considerably worse if the file we are fetching is dynamically
    // generated with an expensive operation.
    std::string newurl ((proto == "web" ? "http" : proto) + ":" + path);
    const char	*curlopts [] = {
	"curl", "-L", "-f", "--head", "-q", "-s", "--url", newurl.c_str (), 0 };
    StringList	headers (StringOps::split (readAllInput (curlopts), '\n'));
    for (size_t i = 0; i < headers.size (); ++i)
    {
	if (! strncmp (headers [i].c_str (), "HTTP/1.1 404 ", 13))
	    return false;
	else if (! strncmp (headers [i].c_str (), "Content-Length:", 15))
	{
	    if (size) *size = atoll (&headers [i][15]);
	    return true;
	}
    }

    return StorageMaker::check (proto, path, size);
}
