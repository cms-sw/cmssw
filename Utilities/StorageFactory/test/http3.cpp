#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  bool exists = StorageFactory::get ()->check
    ("http://cmsdoc.cern.ch/cms.htmlx");

  std::cout << "exists = " << exists << "\n";
  std::cout << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
