#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"

int main (int argc, char **argv)
{
  initTest();

  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " FILE\n";
    return EXIT_FAILURE;
  }

  IOOffset	size = -1;
  bool		exists = StorageFactory::get ()->check(argv [1], &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  if (! exists) return EXIT_SUCCESS;

  static const int SIZE = 1048576;
  Storage	*s = StorageFactory::get ()->open (argv [1]);
  char		*buf = (char *) malloc (SIZE);
  IOSize	n;

  while ((n = s->read (buf, SIZE)))
    std::cout.write (buf, n);

  s->close();
  delete s;
  free (buf);

  std::cerr << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
