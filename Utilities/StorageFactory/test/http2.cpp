#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  IOOffset	size = -1;
  bool		exists = StorageFactory::get ()->check
    ("http://cmsdoc.cern.ch/cms.html", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
