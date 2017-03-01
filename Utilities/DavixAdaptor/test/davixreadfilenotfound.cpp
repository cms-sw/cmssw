#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/test/Test.h"
#include <cassert>

int main(int, char ** /*argv*/) try {
  initTest();

  char buf[1024];
  std::unique_ptr<Storage> s = StorageFactory::get()->open(
      "http://cern.ch/cmsbuild/cms/mc/this/file/does/not/exist.root");
  assert(s);

  s->read(buf, sizeof(buf));
  s->close();

  std::cout << "stats:\n" << StorageAccount::summaryText() << std::endl;
  return EXIT_FAILURE;
} catch (cms::Exception const &e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_SUCCESS;
} catch (std::exception const &e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
