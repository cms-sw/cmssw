#ifndef SimG4CMS_HFShowerParam_h
#define SimG4CMS_HFShowerParam_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerParam.h
// Description: Generates hits for HF with some parametrized information
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "SimG4CMS/Calo/interface/HFGflash.h"

#include "G4ThreeVector.hh"

class G4Step;

#include <TH1F.h>
#include <TH2F.h>
#include <string>
#include <vector>

class HFShowerParam {
public:
  HFShowerParam(const std::string& name,
                const HcalDDDSimConstants* hcons,
                const HcalSimulationParameters* hps,
                edm::ParameterSet const& p);
  virtual ~HFShowerParam();

public:
  struct Hit {
    Hit() {}
    G4ThreeVector position;
    int depth;
    double time;
    double edep;
  };
  std::vector<Hit> getHits(const G4Step* aStep, double weight, bool& isKilled);

private:
  const HcalDDDSimConstants* hcalConstants_;
  std::unique_ptr<HFShowerLibrary> showerLibrary_;
  std::unique_ptr<HFFibre> fibre_;
  std::unique_ptr<HFGflash> gflash_;
  bool fillHisto_;
  double pePerGeV_, edMin_, ref_index_, aperture_, attLMeanInv_;
  bool trackEM_, onlyLong_, applyFidCut_, parametrizeLast_;
  G4int emPDG_, epPDG_, gammaPDG_;
  std::vector<double> gpar_;
  TH1F *em_long_1_, *em_lateral_1_, *em_long_2_, *em_lateral_2_;
  TH1F *hzvem_, *hzvhad_, *em_long_1_tuned_, *em_long_gflash_;
  TH1F* em_long_sl_;
  TH2F *em_2d_1_, *em_2d_2_;
};

#endif  // HFShowerParam_h
