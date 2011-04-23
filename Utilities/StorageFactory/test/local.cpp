#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"

int main (int, char **/*argv*/)
{
  initTest();

  Storage	*s = StorageFactory::get ()->open ("/etc/passwd");
  char		buf [1024];
  IOSize	n;

  while ((n = s->read (buf, sizeof (buf))))
	std::cout.write (buf, n);

  s->close();
  delete s;

  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
