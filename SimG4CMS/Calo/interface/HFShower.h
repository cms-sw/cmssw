#ifndef SimG4CMS_HFShower_h
#define SimG4CMS_HFShower_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShower.h
// Description: Generates hits for HF with Cerenkov photon code
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "G4ThreeVector.hh"
#include "G4String.hh"

class G4Step;

#include <vector>

class HFShower {
public:
  HFShower(const std::string &name,
           const HcalDDDSimConstants *hcons,
           const HcalSimulationParameters *hps,
           edm::ParameterSet const &p,
           int chk = 0);
  virtual ~HFShower();

public:
  struct Hit {
    Hit() {}
    int depth;
    double time;
    double wavelength;
    double momentum;
    G4ThreeVector position;
  };

  std::vector<Hit> getHits(const G4Step *aStep, double weight);
  std::vector<Hit> getHits(const G4Step *aStep, bool forLibrary);
  std::vector<Hit> getHits(const G4Step *aStep, bool forLibraryProducer, double zoffset);

private:
  const HcalDDDSimConstants *hcalConstant_;

  std::unique_ptr<HFCherenkov> cherenkov_;
  std::unique_ptr<HFFibre> fibre_;

  int chkFibre_;
  bool applyFidCut_;
  bool ignoreTimeShift_;
  double probMax_;
  std::vector<double> gpar_;
};

#endif  // HFShower_h
