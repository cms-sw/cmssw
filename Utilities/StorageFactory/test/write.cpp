#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>

int main (int, char **/*argv*/)
{
  initTest();

  try
  {
    struct passwd *info = getpwuid (getuid());
    std::string user (info && info->pw_name ? info->pw_name : "unknown");
    std::string	path (std::string ("rfio:/castor/cern.ch/user/")
    	       	      + user[0] + "/" + user + "/rfiotest");
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
  }
  catch (cms::Exception &e)
  {
    std::cerr << e.explainSelf () << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << StorageAccount::summaryXML() << std::endl;
  return EXIT_SUCCESS;
}
