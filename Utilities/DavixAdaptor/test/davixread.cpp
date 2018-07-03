#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/test/Test.h"
#include <cassert>

int main(int, char ** /*argv*/) try {
  initTest();

  IOSize n;
  IOSize size = 1024;
  char buf[size];
  std::unique_ptr<Storage> s = StorageFactory::get()->open(
      "http://cern.ch/cmsbuild/cms/Run2011A/PhotonHad/AOD"
      "/12Oct2013-v1/00000/024938EB-3445-E311-A72B-002590593920.root");
  assert(s);

  if ((n = s->read(buf, sizeof(buf)) != size)) {
    std::cout << "Requested: " << size << " Got: " << n;
    s->close();
    std::cout << "stats:\n" << StorageAccount::summaryText() << std::endl;
    return EXIT_FAILURE;
  }
  s->close();
  std::cout << "stats:\n" << StorageAccount::summaryText() << std::endl;
  return EXIT_SUCCESS;
} catch (cms::Exception const &e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch (std::exception const &e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
