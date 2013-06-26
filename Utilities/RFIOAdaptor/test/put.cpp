#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <string>

int main (int argc, char *argv[])
{
  try
  {
    std::string	path ("/castor/cern.ch/cms/test/IBTestFiles/rfiotestput");
    if (argc > 1) {
      path = argv[1];
    }
    std::cout << "copying /etc/motd to " << path << "\n";

    File	input ("/etc/profile");
    RFIOFile	output (path.c_str (),
    		        IOFlags::OpenRead
    		        | IOFlags::OpenWrite
    		        | IOFlags::OpenCreate
    		        | IOFlags::OpenTruncate);

    unsigned char buf [4096];
    IOSize	bytes;

    while ((bytes = input.read (buf, sizeof (buf))))
        output.write (buf, bytes);

    input.close ();
    output.close ();
  }
  catch (cms::Exception &e)
  {
    std::cerr << e.explainSelf () << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
