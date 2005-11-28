#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
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
int main (int argc, char **argv)
{
    Signal::handleFatal (argv [0]);
    PluginManager::get ()->initialise ();

    if (argc < 2)
    {
	std::cerr << " please give file name" <<std::endl;
	return EXIT_FAILURE;
    }

    IOOffset    size = -1;
    bool	exists = StorageFactory::get ()->check(argv [1], &size);
    std::cerr << "exists = " << exists << ", size = " << size << "\n";
    if (! exists) return EXIT_SUCCESS;

    Storage	*s = StorageFactory::get ()->open (argv [1]);
    char	buf [1024];
    IOSize	n;

    while ((n = s->read (buf, sizeof (buf))))
	std::cout.write (buf, n);

    delete s;

    std::cerr << "stats:\n" << StorageAccount::summaryText ();
    return EXIT_SUCCESS;
}
