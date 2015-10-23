#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/File.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <string>

int main (int argc, char *argv[]) try
{
  initTest();

    std::string	path ("rfio:/castor/cern.ch/cms/test/IBTestFiles/rfiotestwrite");
    if (argc > 1) {
      path = std::string ("rfio:") + argv[1];
    }
    std::cout << "copying /etc/profile to " << path << "\n";

    IOSize		bytes;
    unsigned char	buf [4096];
    File		input ("/etc/profile");
    auto s = StorageFactory::get ()->open
      (path.c_str (), IOFlags::OpenWrite|IOFlags::OpenCreate|IOFlags::OpenTruncate);

    while ((bytes = input.read (buf, sizeof (buf))))
        s->write (buf, bytes);

    input.close ();
    s->close ();

  std::cout << StorageAccount::summaryText(true) << std::endl;
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
