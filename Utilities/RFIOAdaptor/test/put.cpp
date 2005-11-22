//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/Configuration/interface/Architecture.h"
#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "SealBase/File.h"
#include "SealBase/Error.h"
#include "SealBase/Signal.h"
#include <iostream>
#include <iomanip>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

using namespace seal;
int main (int, char **argv)
{
    Signal::handleFatal (argv [0]);

    try
    {
	std::string	user (UserInfo::self ()->id ());
	std::string	path (std::string ("/castor/cern.ch/user/")
		       	      + user[0] + "/" + user + "/rfiotest");
	std::cout << "copying /etc/motd to " << path << "\n";

	File		input ("/etc/profile");
	RFIOFile	output (path.c_str (),
			        IOFlags::OpenRead
			        | IOFlags::OpenWrite
			        | IOFlags::OpenCreate
			        | IOFlags::OpenTruncate);

	unsigned char	buf [4096];
	IOSize		bytes;

	while ((bytes = input.read (buf, sizeof (buf))))
	    output.write (buf, bytes);

	input.close ();
	output.close ();
    }
    catch (Error &e)
    {
	std::cerr << e.explain () << std::endl;
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
