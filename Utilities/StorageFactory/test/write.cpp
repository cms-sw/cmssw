#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <string>

int main (int argc, char *argv[]) try
{
  initTest();

    struct passwd *info = getpwuid (getuid());
    std::string user (info && info->pw_name ? info->pw_name : "unknown");
    std::string	path (std::string ("rfio:/castor/cern.ch/user/")
    	       	      + user[0] + "/" + user + "/rfiotest");
    if (argc > 1) {
      std::string scramArch(argv[1]);
      path += scramArch;
    }
    std::cout << "copying /etc/profile to " << path << "\n";

    IOSize		bytes;
    unsigned char	buf [4096];
    File		input ("/etc/profile");
    Storage		*s = StorageFactory::get ()->open
      (path.c_str (), IOFlags::OpenWrite|IOFlags::OpenCreate|IOFlags::OpenTruncate);

    while ((bytes = input.read (buf, sizeof (buf))))
        s->write (buf, bytes);

    input.close ();
    s->close ();

  std::cout << StorageAccount::summaryXML() << std::endl;
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
