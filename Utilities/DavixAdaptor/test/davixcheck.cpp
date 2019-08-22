#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/test/Test.h"

int main(int, char ** /*argv*/) try {
  initTest();

  IOOffset size = -1;
  bool exists = StorageFactory::get()->check(
      "http://opendata.cern.ch/eos/opendata"
      "/cms/Run2011A/PhotonHad/AOD/12Oct2013-v1"
      "/00000/024938EB-3445-E311-A72B-002590593920.root",
      &size);

  std::cout << "exists = " << exists << ", size = " << size << "\n";
  std::cout << "stats:\n" << StorageAccount::summaryText() << std::endl;
  return EXIT_SUCCESS;
} catch (cms::Exception const &e) {
  std::cerr << e.explainSelf() << std::endl;
  return EXIT_FAILURE;
} catch (std::exception const &e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
