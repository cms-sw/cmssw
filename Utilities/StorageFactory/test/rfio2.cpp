#include "Utilities/StorageFactory/test/Test.h"
#include "FWCore/Utilities/interface/Exception.h"

int main (int, char **/*argv*/) try
{
  initTest();

  IOOffset	size;
  bool		exists = StorageFactory::get ()->check
    ("rfio:/castor/cern.ch/cms/reconstruction/datafiles/"
     "ORCA_7_5_2/PoolFileCatalog.xml", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << StorageAccount::summaryXML() << std::endl;
  return EXIT_SUCCESS;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
