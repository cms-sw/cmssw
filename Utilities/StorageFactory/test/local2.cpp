#include "Utilities/StorageFactory/test/Test.h"

int main (int, char **argv)
{
  initTest();

  IOOffset	size;
  bool		exists = StorageFactory::get ()->check ("/etc/passwd", &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
