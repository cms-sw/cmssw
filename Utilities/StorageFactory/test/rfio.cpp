#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <cassert>

int main (int, char **/*argv*/)
{
  initTest();

  IOSize	n;
  char		buf [1024];
  Storage	*s = StorageFactory::get ()->open
    ("rfio:/castor/cern.ch/cms/reconstruction/datafiles/ORCA_7_5_2/PoolFileCatalog.xml");

  assert (s);
  while ((n = s->read (buf, sizeof (buf))))
    std::cout.write (buf, n);

  s->close();
  delete s;

  std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
