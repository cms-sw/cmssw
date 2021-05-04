#ifndef SimG4CMS_HFShowerFibreBundle_h
#define SimG4CMS_HFShowerFibreBundle_h
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerFibreBundle.h
// Description: Get energy deposits for HFShower PMT's
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "SimG4CMS/Calo/interface/HFCherenkov.h"

class G4Step;

#include <string>
#include <vector>

class HFShowerFibreBundle {
public:
  HFShowerFibreBundle(const std::string &name,
                      const HcalDDDSimConstants *hcons,
                      const HcalSimulationParameters *hps,
                      edm::ParameterSet const &p);
  virtual ~HFShowerFibreBundle();
  double getHits(const G4Step *aStep, bool type);
  double getRadius();

private:
  const HcalDDDSimConstants *hcalConstant_;
  const HcalSimulationParameters *hcalsimpar_;
  std::unique_ptr<HFCherenkov> cherenkov1_, cherenkov2_;
  double facTube, facCone;  //Packing factors
  int indexR, indexF;
  std::vector<double> rTable;       // R-table
  std::vector<int> pmtR1, pmtFib1;  // R-index, fibre table for right box
  std::vector<int> pmtR2, pmtFib2;  // R-index, fibre table for left box
};

#endif  // HFShowerFibreBundle_h
