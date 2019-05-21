#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/test/Test.h"
#include <cassert>

int main(int, char ** /*argv*/) try {
  initTest();

  char buf[1024];
  std::unique_ptr<Storage> s = StorageFactory::get()->open(
      "http://i.am.davix.plugin.fake.url:1234/store/mc/HC/GenericTTbar/ "
      "GEN/127938CD-F8CC-E311-9250-02163E00E8E6.root");
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
