#ifndef SimG4CMS_HFShowerPMT_h
#define SimG4CMS_HFShowerPMT_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPMT.h
// Description: Maps HF PMT's to given
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"

class G4Step;

#include <string>
#include <vector>

class HFShowerPMT {
public:
  HFShowerPMT(const std::string& name,
              const HcalDDDSimConstants* hcons,
              const HcalSimulationParameters* hps,
              edm::ParameterSet const& p);
  virtual ~HFShowerPMT();
  double getHits(const G4Step* aStep);
  double getRadius();

private:
  const HcalDDDSimConstants* hcalConstant_;
  const HcalSimulationParameters* hcalsimpar_;
  std::unique_ptr<HFCherenkov> cherenkov_;
  double pePerGeV;  // PE per GeV of energy deposit
  int indexR, indexF;
  std::vector<double> rTable;       // R-table
  std::vector<int> pmtR1, pmtFib1;  // R-index, fibre table for right box
  std::vector<int> pmtR2, pmtFib2;  // R-index, fibre table for left box
};

#endif  // HFShowerPMT_h
