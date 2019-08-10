#ifndef SimG4CMS_HFFibre_h
#define SimG4CMS_HFFibre_h 1
///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.h
// Description: Calculates attenuation length
///////////////////////////////////////////////////////////////////////////////

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "G4ThreeVector.hh"

#include <vector>
#include <string>

class HFFibre {
public:
  //Constructor and Destructor
  HFFibre(const std::string& name, const HcalDDDSimConstants* hcons, edm::ParameterSet const& p);
  ~HFFibre() = default;

  double attLength(double lambda);
  double tShift(const G4ThreeVector& point, int depth, int fromEndAbs = 0);
  double zShift(const G4ThreeVector& point, int depth, int fromEndAbs = 0);

private:
  const HcalDDDSimConstants* hcalConstant_;
  double cFibre;
  std::vector<double> gpar, radius;
  std::vector<double> shortFL, longFL;
  std::vector<double> attL;
  int nBinR, nBinAtt;
  double lambLim[2];
};
#endif
