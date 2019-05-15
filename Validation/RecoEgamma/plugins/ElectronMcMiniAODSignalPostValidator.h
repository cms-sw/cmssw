
#ifndef Validation_RecoEgamma_ElectronMcSignalPostValidatorMiniAOD_h
#define Validation_RecoEgamma_ElectronMcSignalPostValidatorMiniAOD_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"

class ElectronMcSignalPostValidatorMiniAOD : public ElectronDqmHarvesterBase {
public:
  explicit ElectronMcSignalPostValidatorMiniAOD(const edm::ParameterSet& conf);
  ~ElectronMcSignalPostValidatorMiniAOD() override;
  void finalize(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;

private:
  std::string inputFile_;
  std::string outputFile_;
  std::vector<int> matchingIDs_;
  std::vector<int> matchingMotherIDs_;
  std::string inputInternalPath_;
  std::string outputInternalPath_;

  // histos limits and binning
  bool set_EfficiencyFlag;
  bool set_StatOverflowFlag;

  // histos
  //    MonitorElement *h1_ele_xOverX0VsEta ;
};

#endif
