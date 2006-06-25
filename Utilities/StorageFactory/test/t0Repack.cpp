#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealIOTools/StorageStreamBuf.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>



using namespace seal;
int main (int argc, char **argv)
{

    Signal::handleFatal (argv [0]);
    PluginManager::get ()->initialise ();

    if (argc < 4)
    {
	std::cerr << "please give dataset number, output file name, and list of index files" <<std::endl;
	return EXIT_FAILURE;
    }

    StorageFactory::get ()->enableAccounting(true);

    int datasetN = ::atol(argv [1]);
    std::string outputURL = argv[2];
    std::cerr << "write to file " << outputURL
	      << " dataset " << datasetN << std::endl;
    std::vector<Storage	*> indexFiles;
    std::vector<IOOffset> indexSizes;
    for ( int i=3; i<argc; i++ ) {
      IOOffset    size = -1;
      if (StorageFactory::get ()->check(argv [i], &size)) {	
	indexFiles.push_back(StorageFactory::get ()->
			     open (argv [i],seal::IOFlags::OpenRead));
	indexSizes.push_back(size);
      }
      else {
	std::cerr << "index file " << argv [i] << " does not exists" << std::endl;
	return EXIT_FAILURE;
      }
    }

    // open output file
    Storage  * outputFile = 0;
    try {
      outputFile = StorageFactory::get ()->open (outputURL,
						 IOFlags::OpenWrite
						 | IOFlags::OpenCreate
						 | IOFlags::OpenTruncate);
    } catch (...) {
      std::cerr << "error in opening output file " << outputURL << std::endl;
      return EXIT_FAILURE;
    }

    // parse index file
    // read buffer
    // select and copy to output file
    IOSize totSize=0;
    for (unsigned int i=0; i<indexFiles.size();i++) {
      std::cerr << "reading from index file " <<  argv[i+3] << std::endl;
      //StorageStreamBuf	bufio (indexFiles[i]);
      // std::istream in (&bufio);
      // std::ifstream in(argv [i+3]);
      // get the whole file in memory
      std::istringstream in;
      try {
	std::vector<char> lbuf(indexSizes[i]+1,'\0');
	IOSize nn = indexFiles[i]->read(&lbuf[0],indexSizes[i]);
	if (nn!=indexSizes[i]) {
	      std::cerr << "error in reading from  index file " <<  argv[i+3] << std::endl;
	      std::cerr << "asked for " <<  indexSizes[i] <<". got " << nn << std::endl;
	      return EXIT_FAILURE;
	}
	in.str(&lbuf[0]);
      } catch (...) {
	std::cerr << "error in reading from index file " << argv [i+3] << std::endl;
	return EXIT_FAILURE;
      
      }

	std::string line1; std::getline(in, line1);
	std::cerr << "first line is:\n" << line1 << std::endl;
	std::string::size_type pos = line1.find('=');
      if (pos!=std::string::npos) pos = line1.find_first_not_of(' ',pos+1);
      if (pos==std::string::npos) {
	std::cerr << "badly formed index file " << argv [i+3] << std::endl;
	std::cerr << "first line is:\n" << line1 << std::endl;
	return EXIT_FAILURE;
      }
      line1.erase(0,pos);
      std::cerr << "input event file " << i << " is " << line1 << std::endl;
      Storage	* s; 
      IOOffset size=0;
      try {
	if (StorageFactory::get ()->check(line1, &size)) {	
	 s = StorageFactory::get ()->
	   open (line1,seal::IOFlags::OpenRead);
	}
	else {
	  std::cerr << "input file " << line1 << " does not exists" << std::endl;
	  return EXIT_FAILURE;
	}
      } catch (...) {
	std::cerr << "error in opening input file " << line1 << std::endl;
	  return EXIT_FAILURE;
      }
      //
      std::vector<char> vbuf;
      while(in) {
	int dataset=-1;
	IOOffset bufLoc = -1;
	IOSize   bufSize = 0;
	in >> dataset >> bufLoc >> bufSize;
	if (dataset==datasetN) {
	  std::cerr << "copy buf at " << bufLoc << " of size " << bufSize << std::endl;
	  if (bufSize>vbuf.size()) vbuf.resize(bufSize);
	  char * buf = &vbuf[0];
	  try {
	    s->position(bufLoc);
	    IOSize  n = s->read (buf, bufSize);
	    totSize+=n;
	    if (n!= bufSize) {
	      std::cerr << "error in reading from  input file " << line1 << std::endl;
	      std::cerr << "asked for " << bufSize <<". got " << n << std::endl;
	      return EXIT_FAILURE;
	    }
	  } catch (...) {
	    std::cerr << "error in reading from  input file " << line1 << std::endl;
	    return EXIT_FAILURE;
	  }
	  try {
	    outputFile->write(buf,bufSize);
	  } catch (...) {
	    std::cerr << "error in writing to output file " << outputURL << std::endl;
	    return EXIT_FAILURE;
	  }
	}
      }

      delete s;
 
    }

    outputFile->close();
    delete  outputFile;

    std::cerr << "copied a total of " << totSize << " bytes" << std::endl;


    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;

    return EXIT_SUCCESS;
}

