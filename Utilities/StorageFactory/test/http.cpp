#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <cassert>

int main (int, char **/*argv*/)
{
  initTest();

  IOSize	n;
  char		buf [1024];
  Storage	*s = StorageFactory::get ()->open
    ("http://cmsdoc.cern.ch/cms.html", IOFlags::OpenRead);

  assert (s);
  while ((n = s->read (buf, sizeof (buf))))
    std::cout.write (buf, n);

  s->close();
  delete s;

  std::cerr << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
