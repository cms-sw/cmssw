#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/test/Test.h"
#include <cassert>

int main(int, char ** /*argv*/) try {
  initTest();

  std::unique_ptr<Storage> s = StorageFactory::get()->open(
      "http://cern.ch/cmsbuild/cms/Run2011A/PhotonHad/AOD"
      "/12Oct2013-v1/00000/024938EB-3445-E311-A72B-002590593920.root");
  assert(s);

  IOSize totalVecs = 6;
  char *buf[4096] = {NULL};
  int sizes[] = {20, 100, 50, 1024, 2222};
  int offset[] = {1000, 2, 19, 100, 500};

  for (IOSize i = 1; i < totalVecs; i++) {
    std::cout << i;
    std::vector<IOPosBuffer> iov;
    iov.reserve(i);
    for (IOSize j = 0; j < i; j++) {
      iov.push_back(IOPosBuffer(offset[j], buf, sizes[j]));
    }
    s->readv(&iov[0], iov.size());
    std::cout << "stats " << i << ":\n" << StorageAccount::summaryText() << std::endl;
  }

  s->close();

  std::cout << "final stats:\n" << StorageAccount::summaryText() << std::endl;
  return EXIT_SUCCESS;
} catch (cms::Exception const &e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch (std::exception const &e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
