#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

int main (int, char **/*argv*/) try
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
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
