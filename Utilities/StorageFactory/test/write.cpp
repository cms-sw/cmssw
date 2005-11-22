//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/Configuration/interface/Architecture.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include "SealBase/Error.h"
#include "SealBase/File.h"
#include <iostream>

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
    PluginManager::get ()->initialise ();

    try
    {
	std::string	user (UserInfo::self ()->id ());
	std::string	path (std::string ("rfio:/castor/cern.ch/user/")
		       	      + user[0] + "/" + user + "/rfiotest");
	std::cout << "copying /etc/profile to " << path << "\n";

	IOSize		bytes;
	unsigned char	buf [4096];
	File		input ("/etc/profile");
    	Storage		*s = StorageFactory::get ()->open (path.c_str (),
							   IOFlags::OpenWrite
							   | IOFlags::OpenCreate
							   | IOFlags::OpenTruncate);

	while ((bytes = input.read (buf, sizeof (buf))))
	    s->write (buf, bytes);

	input.close ();
	s->close ();
    }
    catch (Error &e)
    {
	std::cerr << e.explain () << std::endl;
	return EXIT_FAILURE;
    }

    std::cerr << "stats:\n" << StorageAccount::summaryText ();
    return EXIT_SUCCESS;
}
