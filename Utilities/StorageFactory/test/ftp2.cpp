#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  IOOffset	size = -1;
  bool		exists = StorageFactory::get ()->check
    ("ftp://cmsdoc.cern.ch/non-existent", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
