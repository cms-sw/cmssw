#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <cmath>
#include <limits>
#include <vector>

int main (int argc, char **argv)
{
  initTest();

  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " FILE...\n";
    return EXIT_FAILURE;
  }

  StorageFactory::get ()->enableAccounting(true);
  std::vector<Storage *> storages;
  std::vector<IOOffset> sizes;	
  for (int i = 1; i < argc; ++i)
  {
    IOOffset    size = -1;
    bool	exists = StorageFactory::get ()->check(argv [i], &size);

    std::cout << argv[i] << " exists = " << exists << ", size = " << size << std::endl;

    if (exists)
    {	
      storages.push_back(StorageFactory::get ()->open
			 (argv [i],IOFlags::OpenRead|IOFlags::OpenUnbuffered));
      sizes.push_back(size);   
    }
  }

  if (sizes.empty())
    return EXIT_SUCCESS;

  std::cout << "stats:\n" << StorageAccount::summaryXML () << std::endl;

  IOSize	n;
  char		buf [10000];
  IOSize	maxBufSize = sizeof(buf);
  int		niter = 100000;

  while (niter--) 
    for (size_t i = 0; i < sizes.size(); ++i)
    {
      double bfract = double(rand()) / double(std::numeric_limits<int>::max());
      double pfract = double(rand()) / double(std::numeric_limits<int>::max());
      IOSize bufSize = static_cast<IOSize>(maxBufSize * bfract);
      IOOffset pos = static_cast<IOOffset>((sizes[i] - bufSize)*pfract);
      // std::cout << "read " << bufSize << " at " << pos << std::endl;
      storages[i]->position(pos);	
      n = storages[i]->read (buf, bufSize);
      if (n != bufSize)
      {
	std::cerr << "error for " << i << " (" << argv[i+1]
		  << "): tried to read " << bufSize << " bytes at " << pos
		  << "; got " << n << " bytes\n";
    	break;	
      }
    }

  for (size_t i = 0; i < sizes.size(); ++i) 
  {
    storages[i]->close();
    delete storages[i];
  }

  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
