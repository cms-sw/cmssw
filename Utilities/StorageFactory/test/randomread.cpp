#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include <iostream>
#include <math>
#include <limits>

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
    StorageFactory::get ()->enableAccounting(true);
    bool	exists = StorageFactory::get ()->check(argv [1], &size);
    std::cerr << "exists = " << exists << ", size = " << size << std::endl;
    if (! exists) return EXIT_SUCCESS;

    Storage	*s = StorageFactory::get ()->open (argv [1],seal::IOFlags::OpenRead|seal::IOFlags::OpenUnbuffered);
    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    char	buf [10000];
    IOSize	n;
	IOSize maxBufSize = sizeof(buf);
	int n_iter=100000;
    while (n_iter--) {
    	IOSize bufSize = maxBufSize*(double(rand())/
    		double(std::numeric_limits<unsigned int>::max()));
    	IOOffset l_pos = (size-bufSize)*(double(rand())/
    		double(std::numeric_limits<unsigned int>::max()));
    	// std::cout << "read " << bufSize << " at " << l_pos << std::endl;
	    s->position(l_pos);	
    	n = s->read (buf, bufSize);
		if (n!=bufSize) {
			std::cerr << "error: try to read " << n 
					<< " at " << l_pos << "; got " << n << std::endl;
			break;	
		}
	}
    delete s;


    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;

    return EXIT_SUCCESS;
}
