
#ifndef Validation_RecoEgamma_ElectronMcFakePostValidator_h
#define Validation_RecoEgamma_ElectronMcFakePostValidator_h

#include "DQMOffline/EGamma/interface/ElectronDqmHarvesterBase.h"

class ElectronMcFakePostValidator : public ElectronDqmHarvesterBase {
public:
  explicit ElectronMcFakePostValidator(const edm::ParameterSet &conf);
  ~ElectronMcFakePostValidator() override;
  //    virtual void book() ;
  void finalize(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) override;  //

private:
  std::string inputFile_;
  std::string outputFile_;
  std::string inputInternalPath_;
  std::string outputInternalPath_;

  // histos limits and binning
  bool set_EfficiencyFlag;
  bool set_StatOverflowFlag;

  // histos
  MonitorElement *h1_ele_xOverX0VsEta;
};

#endif
