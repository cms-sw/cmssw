#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/StorageFactory/test/Test.h"

int main(int, char ** /*argv*/) try {
  initTest();

  IOOffset size = -1;
  bool exists = StorageFactory::get()->check("http://transfer-8.ultralight.org:1094/store/mc/HC/"
                                             "GenericTTbar/GEN-SIM-RECO/CMSSW_7_0_4_START70_V7-v1/"
                                             "00000///127938CD-F8CC-E311-9250-02163E00E8E6.root",
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
