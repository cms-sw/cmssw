#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
# include <boost/shared_ptr.hpp>
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

    if (argc <3)
    {
	std::cerr << " please give input and output file names" <<std::endl;
	return EXIT_FAILURE;
    }

    IOOffset    size = -1;
    StorageFactory::get ()->enableAccounting(true);
    bool	exists = StorageFactory::get ()->check(argv [1], &size);
    std::cerr << "exists = " << exists << ", size = " << size << "\n";
    if (! exists) return EXIT_SUCCESS;

     boost::shared_ptr<Storage> is;

    try {
      is.reset(StorageFactory::get ()->open (argv [1]));
    } catch (...) {
      std::cerr << "error in opening input file " << argv[1] << std::endl;
      return EXIT_FAILURE;
    }



    // open output file
     boost::shared_ptr<Storage> os;
    try {
      os.reset(StorageFactory::get ()->open (argv[2],
					     IOFlags::OpenWrite
					     | IOFlags::OpenCreate
					     | IOFlags::OpenTruncate)
	       );
    } catch (...) {
      std::cerr << "error in opening output file " << argv[2] << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<char> buf(100000);
    IOSize	n;

    while ((n = is->read (&buf[0], buf.size())))
      os->write (&buf[0], n);


    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    return EXIT_SUCCESS;
}
