#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/Exception.h"

int main (int, char **/*argv*/) try
{
  initTest();

  auto s = StorageFactory::get ()->open ("/etc/passwd");
  char		buf [1024];
  IOSize	n;

  while ((n = s->read (buf, sizeof (buf))))
	std::cout.write (buf, n);

  s->close();

  std::cout << StorageAccount::summaryText (true) << std::endl;
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
