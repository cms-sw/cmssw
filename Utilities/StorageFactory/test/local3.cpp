#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **/*argv*/)
{
  initTest();

  bool exists = StorageFactory::get ()->check ("/etc/passwdx");
  std::cout << "exists = " << exists << "\n";
  std::cout << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
