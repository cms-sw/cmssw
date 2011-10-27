#include "Utilities/RFIOAdaptor/interface/RFIOFile.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <iostream>
#include <iomanip>
#include <string>

int main (int argc, char *argv[])
{
  try
  {
    struct passwd *info = getpwuid (getuid());
    std::string	user (info && info->pw_name ? info->pw_name : "unknown");
    std::string	path (std::string ("/castor/cern.ch/user/")
    	       	      + user[0] + "/" + user + "/rfiotestput");
    if (argc > 1) {
      std::string scramArch(argv[1]);
      path += scramArch;
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
