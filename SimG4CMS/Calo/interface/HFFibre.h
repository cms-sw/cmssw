#ifndef SimG4CMS_HFFibre_h
#define SimG4CMS_HFFibre_h 1
///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.h
// Description: Calculates attenuation length
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"

#include "G4ThreeVector.hh"

#include <vector>
#include <string>
#include <array>

class HFFibre {
public:
  //Constructor and Destructor
  HFFibre(const HcalDDDSimConstants* hcons, const HcalSimulationParameters* hps, edm::ParameterSet const& p);

  double attLength(double lambda) const;
  double tShift(const G4ThreeVector& point, int depth, int fromEndAbs = 0) const;
  double zShift(const G4ThreeVector& point, int depth, int fromEndAbs = 0) const;

  struct Params {
    Params() = default;
    Params(double iFractionOfSpeedOfLightInFibre,
           const HcalDDDSimConstants* hcons,
           const HcalSimulationParameters* hps);
    double fractionOfSpeedOfLightInFibre_;
    std::vector<double> gParHF_;
    std::vector<double> rTableHF_;
    std::vector<double> shortFibreLength_;
    std::vector<double> longFibreLength_;
    std::vector<double> attenuationLength_;
    std::array<double, 2> lambdaLimits_;
  };

  HFFibre(Params iP);

private:
  double cFibre_;
  std::vector<double> gpar_, radius_;
  std::vector<double> shortFL_, longFL_;
  std::vector<double> attL_;
  int nBinR_, nBinAtt_;
  std::array<double, 2> lambLim_;
};
#endif
