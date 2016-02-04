#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <memory>

int main (int argc, char **argv)
{
  initTest();

  if (argc < 4)
  {
    std::cerr << "usage: " << argv[0] << " NUM-DATASETS OUTPUT-FILE INDEX-FILE...\n";
    return EXIT_FAILURE;
  }

  int			 datasetN = ::atoi(argv [1]);
  std::string		 outputURL = argv[2];
  Storage		 *outputFile = 0;
  IOSize		 totSize = 0;
  std::vector<Storage *> indexFiles;
  std::vector<IOOffset>  indexSizes;

  StorageFactory::get ()->enableAccounting(true);
  std::cerr << "write to file " << outputURL << " " << datasetN << " datasets\n";

  for (int i = 3; i < argc; ++i)
  {
    IOOffset size = -1;
    if (StorageFactory::get ()->check(argv [i], &size))
    {
      indexFiles.push_back(StorageFactory::get ()->open (argv [i],IOFlags::OpenRead));
      indexSizes.push_back(size);
    }
    else
    {
      std::cerr << "index file " << argv [i] << " does not exist\n";
      return EXIT_FAILURE;
    }
  }

  // open output file
  try
  {
    outputFile = StorageFactory::get ()->open
      (outputURL, IOFlags::OpenWrite|IOFlags::OpenCreate|IOFlags::OpenTruncate);
  }
  catch (cms::Exception &e)
  {
    std::cerr << "error in opening output file " << outputURL
	      << ": " << e.explainSelf () << std::endl;
    return EXIT_FAILURE;
  }

  // parse index file, read buffer, select and copy to output file
  for (size_t i = 0; i < indexFiles.size(); ++i)
  {
    // suck in the index file.
    std::cout << "reading from index file " <<  argv[i+3] << std::endl;
    std::istringstream in;
    try
    {
      std::vector<char> lbuf(indexSizes[i]+1, '\0');
      IOSize		nn = indexFiles[i]->read(&lbuf[0], indexSizes[i]);

      if (indexSizes[i] < 0 || static_cast<IOOffset>(nn) != indexSizes[i])
      {
        std::cerr << "error in reading from index file " <<  argv[i+3] << std::endl;
        std::cerr << "asked for " <<  indexSizes[i] << " bytes, got " << nn << " bytes\n";
        return EXIT_FAILURE;
      }

      in.str(&lbuf[0]);
    }
    catch (cms::Exception &e)
    {
      std::cerr << "error in reading from index file " << argv [i+3] << std::endl
		<< e.explainSelf() << std::endl;
      return EXIT_FAILURE;
    }
    catch (...)
    {
      std::cerr << "error in reading from index file " << argv [i+3] << std::endl;
      return EXIT_FAILURE;
    }

    std::string line1;
    std::getline(in, line1);
    std::cout << "first line is '" << line1 << "'\n";
    std::string::size_type pos = line1.find('=');
    if (pos != std::string::npos)
      pos = line1.find_first_not_of(' ',pos+1);
    if (pos == std::string::npos)
    {
      std::cerr << "badly formed index file " << argv [i+3] << std::endl;
      std::cerr << "first line is:\n" << line1 << std::endl;
      return EXIT_FAILURE;
    }
    line1.erase(0,pos);

    Storage	*s = 0; 
    IOOffset	size = 0;
    try
    {
      std::cout << "input event file " << i << " is " << line1 << std::endl;
      if (StorageFactory::get ()->check(line1, &size))
        s = StorageFactory::get ()->open (line1, IOFlags::OpenRead);
      else
      {
        std::cerr << "input file " << line1 << " does not exist\n";
        return EXIT_FAILURE;
      }
    }
    catch (cms::Exception &e)
    {
      std::cerr << "error in opening input file " << line1 << std::endl
		<< e.explainSelf() << std::endl;
      return EXIT_FAILURE;
    }
    catch (...)
    {
      std::cerr << "error in opening input file " << line1 << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<char> vbuf;
    while (in)
    {
      int	dataset = -1;
      IOOffset	bufLoc = -1;
      IOSize	bufSize = 0;

      in >> dataset >> bufLoc >> bufSize;

      if (dataset != datasetN)
	continue;

      std::cout << "copy buf at " << bufLoc << " of size " << bufSize << std::endl;
      if (bufSize > vbuf.size())
	vbuf.resize(bufSize);

      char * buf = &vbuf[0];
      try
      {
        s->position(bufLoc);
        IOSize n = s->read (buf, bufSize);
        totSize += n;
        if (n != bufSize)
        {
          std::cerr << "error in reading from input file " << line1 << std::endl;
          std::cerr << "asked for " <<  bufSize << " bytes, got " << n << " bytes\n";
          return EXIT_FAILURE;
        }
      }
      catch (cms::Exception &e)
      {
        std::cerr << "error in reading input file " << line1 << std::endl
		  << e.explainSelf() << std::endl;
        return EXIT_FAILURE;
      }
      catch (...)
      {
        std::cerr << "error in reading input file " << line1 << std::endl;
        return EXIT_FAILURE;
      }

      try
      {
	outputFile->write(buf,bufSize);
      }
      catch (cms::Exception &e)
      {
        std::cerr << "error in writing output file " << outputURL << std::endl
		  << e.explainSelf() << std::endl;
        return EXIT_FAILURE;
      }
      catch (...)
      {
        std::cerr << "error in writing output file " << outputURL << std::endl;
        return EXIT_FAILURE;
      }
    }

    s->close();
    delete s;
  }

  outputFile->close();
  delete outputFile;

  std::cout << "copied a total of " << totSize << " bytes" << std::endl;
  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
