#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>

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
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    if (argc < 2)
    {
	std::cerr << " please give file name" <<std::endl;
	return EXIT_FAILURE;
    }

	std::vector<Storage	*> storages;
	std::vector<IOOffset> sizes;	
    StorageFactory::get ()->enableAccounting(true);
    for ( int i=1; i<argc; i++ ) {
	    IOOffset    size = -1;
    	bool	exists = StorageFactory::get ()->check(argv [i], &size);
    	std::cerr << "exists = " << exists << ", size = " << size << std::endl;
    	if (exists) {	
    	 	storages.push_back(StorageFactory::get ()->open (argv [i],seal::IOFlags::OpenRead|	
     						seal::IOFlags::OpenUnbuffered));
     		sizes.push_back(size);   
    	}
    }
    if (sizes.empty()) return EXIT_SUCCESS;

    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    char	buf [10000];
    IOSize	n;
	IOSize maxBufSize = sizeof(buf);
	int n_iter=100000;
    while (n_iter--) 
    for (int i=0;i<sizes.size();i++) {
    	IOSize bufSize = maxBufSize*(double(rand())/
    		double(std::numeric_limits<unsigned int>::max()));
    	IOOffset l_pos = (sizes[i]-bufSize)*(double(rand())/
    		double(std::numeric_limits<unsigned int>::max()));
    	// std::cout << "read " << bufSize << " at " << l_pos << std::endl;
	    storages[i]->position(l_pos);	
    	n = storages[i]->read (buf, bufSize);
		if (n!=bufSize) {
			std::cerr << "error for " << i << ": try to read " << n 
					<< " at " << l_pos << "; got " << n << std::endl;
			break;	
		}
	}
    for (int i=0;i<sizes.size();i++) 
    	delete storages[i];


    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;

    return EXIT_SUCCESS;
}
