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

using namespace edm;

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

HFDarkening::~HFDarkening() { }

double HFDarkening::dose(unsigned int layer, double Radius) {
  // Radii are 13-17, 17-20, 20-24, 24-29, 29-34, 34-41, 41-48, 48-58, 58-69, 69-82, 82-98, 98-116, 116-130
  // These radii are specific to the geometry of the dose map, which closely matches HF Tower Geometry,
  // but not exactly.
  if (layer > (_numberOfZLayers-1))
  {
    return 0.;
  }
  
  int radius = 0;
  if (Radius > 13.0 && Radius <= 17.0) radius = 0;
  if (Radius > 17.0 && Radius <= 20.0) radius = 1;
  if (Radius > 20.0 && Radius <= 24.0) radius = 2;
  if (Radius > 24.0 && Radius <= 29.0) radius = 3;
  if (Radius > 29.0 && Radius <= 34.0) radius = 4;
  if (Radius > 34.0 && Radius <= 41.0) radius = 5;
  if (Radius > 41.0 && Radius <= 48.0) radius = 6;
  if (Radius > 48.0 && Radius <= 58.0) radius = 7;
  if (Radius > 58.0 && Radius <= 69.0) radius = 8;
  if (Radius > 69.0 && Radius <= 82.0) radius = 9;
  if (Radius > 82.0 && Radius <= 98.0) radius = 10;
  if (Radius > 98.0 && Radius <= 116.0) radius = 11;
  if (Radius > 116.0 && Radius <= 130.0) radius = 12;
  if (Radius > 130.0) return 0.;
  
  return HFDoseLayerDarkeningPars[layer][radius];
}

double HFDarkening::degradation(double mrad) {
  return (exp(-1.44*pow(mrad/100,0.44)*0.2/4.343));
}

double HFDarkening::int_lumi(double intlumi) {
  return (intlumi/500.);
}
