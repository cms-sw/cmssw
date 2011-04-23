#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  IOOffset	size;
  bool		exists = StorageFactory::get ()->check
    ("rfio:/castor/cern.ch/cms/reconstruction/datafiles/"
     "ORCA_7_5_2/PoolFileCatalog.xml", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << StorageAccount::summaryXML() << std::endl;
  return EXIT_SUCCESS;
}
