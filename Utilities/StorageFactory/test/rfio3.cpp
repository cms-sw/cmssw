#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  bool exists = StorageFactory::get ()->check
    ("rfio:/castor/cern.ch/cms/reconstruction/datafiles/"
     "ORCA_7_5_2/PoolFileCatalog.xmlx");

  std::cout << "exists = " << exists << "\n";
  std::cout << StorageAccount::summaryXML() << std::endl;
  return EXIT_SUCCESS;
}
