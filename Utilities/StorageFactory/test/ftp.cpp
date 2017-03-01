#include "Utilities/StorageFactory/test/Test.h"
#include "FWCore/Utilities/interface/Exception.h"

int main (int, char **/*argv*/) try
{
  initTest();

  IOOffset	size = -1;
  bool		exists = StorageFactory::get ()->check
    ("ftp://cmsdoc.cern.ch/WELCOME", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << StorageAccount::summaryText(true) << std::endl;
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
