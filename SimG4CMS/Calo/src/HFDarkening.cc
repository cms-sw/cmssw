//////////////////////////////////////////////////////////////////////
//File: HFDarkening.cc
//Description:  simple helper class containing parameterized function
//              to be used for the SLHC darkening calculation in HF
//////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFDarkening.h"
#include <algorithm>
#include <cmath>

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HFDarkening::HFDarkening(const edm::ParameterSet& pset) {
  //HF area of consideration is 1115 cm from interaction point to 1280cm in z-axis
  //Radius (cm) - 13 cm from Beam pipe to 130cm (the top of HF active area)
  //Dose in MRad

  vecOfDoubles HFDosePars = pset.getParameter<vecOfDoubles>("doseLayerDepth");
  int i = 0;
  for (int Z = 0; Z != _numberOfZLayers; ++Z) {
    for (int R = 0; R != _numberOfRLayers; ++R) {
      HFDoseLayerDarkeningPars[Z][R] = HFDosePars[i];
      ++i;
    }
  }
}

HFDarkening::~HFDarkening() {}

double HFDarkening::dose(unsigned int layer, double radius) {
  // Radii are 13-17, 17-20, 20-24, 24-29, 29-34, 34-41, 41-48, 48-58, 58-69, 69-82, 82-98, 98-116, 116-130
  // These radii are specific to the geometry of the dose map, which closely matches HF Tower Geometry,
  // but not exactly.
  if (layer > (_numberOfZLayers - 1)) {
    return 0.;
  }

  int radiusIndex = 0;
  if (radius <= 17.0)
    radiusIndex = 0;
  else if (radius <= 20.0)
    radiusIndex = 1;
  else if (radius <= 24.0)
    radiusIndex = 2;
  else if (radius <= 29.0)
    radiusIndex = 3;
  else if (radius <= 34.0)
    radiusIndex = 4;
  else if (radius <= 41.0)
    radiusIndex = 5;
  else if (radius <= 48.0)
    radiusIndex = 6;
  else if (radius <= 58.0)
    radiusIndex = 7;
  else if (radius <= 69.0)
    radiusIndex = 8;
  else if (radius <= 82.0)
    radiusIndex = 9;
  else if (radius <= 98.0)
    radiusIndex = 10;
  else if (radius <= 116.0)
    radiusIndex = 11;
  else if (radius <= 130.0)
    radiusIndex = 12;
  else
    return 0.;

  return HFDoseLayerDarkeningPars[layer][radiusIndex];
}

double HFDarkening::degradation(double mrad) { return (exp(-1.44 * pow(mrad / 100, 0.44) * 0.2 / 4.343)); }

double HFDarkening::int_lumi(double intlumi) { return (intlumi / 500.); }
