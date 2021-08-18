///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.cc
// Description: Loads the table for attenuation length and calculates it
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <iostream>
#include <sstream>

//#define EDM_ML_DEBUG

HFFibre::HFFibre(const std::string& name,
                 const HcalDDDSimConstants* hcons,
                 const HcalSimulationParameters* hps,
                 edm::ParameterSet const& p)
    : hcalConstant_(hcons), hcalsimpar_(hps) {
  edm::ParameterSet m_HF =
      (p.getParameter<edm::ParameterSet>("HFShower")).getParameter<edm::ParameterSet>("HFShowerBlock");
  cFibre = c_light * (m_HF.getParameter<double>("CFibre"));
  edm::LogVerbatim("HFShower") << "HFFibre:: Speed of light in fibre " << cFibre / (CLHEP::m / CLHEP::ns) << " m/ns";
  // Attenuation length
  attL = hcalsimpar_->attenuationLength_;
  nBinAtt = static_cast<int>(attL.size());
  std::stringstream ss1;
  for (int it = 0; it < nBinAtt; it++) {
    if (it / 10 * 10 == it) {
      ss1 << "\n";
    }
    ss1 << "  " << attL[it] * CLHEP::cm;
  }
  edm::LogVerbatim("HFShower") << "HFFibre: " << nBinAtt << " attL(1/cm): " << ss1.str();
  // Limits on Lambda
  std::vector<int> nvec = hcalsimpar_->lambdaLimits_;
  lambLim[0] = nvec[0];
  lambLim[1] = nvec[1];
  edm::LogVerbatim("HFShower") << "HFFibre: Limits on lambda " << lambLim[0] << " and " << lambLim[1];
  // Fibre Lengths
  longFL = hcalsimpar_->longFiberLength_;
  std::stringstream ss2;
  for (unsigned int it = 0; it < longFL.size(); it++) {
    if (it / 10 * 10 == it) {
      ss2 << "\n";
    }
    ss2 << "  " << longFL[it] / CLHEP::cm;
  }
  edm::LogVerbatim("HFShower") << "HFFibre: " << longFL.size() << " Long Fibre Length(cm):" << ss2.str();
  shortFL = hcalsimpar_->shortFiberLength_;
  std::stringstream ss3;
  for (unsigned int it = 0; it < shortFL.size(); it++) {
    if (it / 10 * 10 == it) {
      ss3 << "\n";
    }
    ss3 << "  " << shortFL[it] / CLHEP::cm;
  }
  edm::LogVerbatim("HFShower") << "HFFibre: " << shortFL.size() << " Short Fibre Length(cm):" << ss3.str();

  // Now geometry parameters
  gpar = hcalConstant_->getGparHF();
  radius = hcalConstant_->getRTableHF();

  nBinR = static_cast<int>(radius.size());
  std::stringstream sss;
  for (int i = 0; i < nBinR; ++i) {
    if (i / 10 * 10 == i) {
      sss << "\n";
    }
    sss << "  " << radius[i] / CLHEP::cm;
  }
  edm::LogVerbatim("HFShower") << "HFFibre: " << radius.size() << " rTable(cm):" << sss.str();
}

double HFFibre::attLength(double lambda) {
  int i = int(nBinAtt * (lambda - lambLim[0]) / (lambLim[1] - lambLim[0]));

  int j = i;
  if (i >= nBinAtt)
    j = nBinAtt - 1;
  else if (i < 0)
    j = 0;
  double att = attL[j];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::attLength for Lambda " << lambda << " index " << i << " " << j
                               << " Att. Length " << att;
#endif
  return att;
}

double HFFibre::tShift(const G4ThreeVector& point, int depth, int fromEndAbs) {
  double zFibre = zShift(point, depth, fromEndAbs);
  double time = zFibre / cFibre;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::tShift for point " << point << " ( depth = " << depth
                               << ", traversed length = " << zFibre / CLHEP::cm << " cm) = " << time / CLHEP::ns
                               << " ns";
#endif
  return time;
}

double HFFibre::zShift(const G4ThreeVector& point, int depth, int fromEndAbs) {  // point is z-local

  double zFibre = 0;
  int ieta = 0;
  double length = 250 * CLHEP::cm;
  double hR = sqrt((point.x()) * (point.x()) + (point.y()) * (point.y()));

  // Defines the Radius bin by radial subdivision
  if (fromEndAbs >= 0) {
    for (int i = nBinR - 1; i > 0; --i)
      if (hR < radius[i])
        ieta = nBinR - i - 1;
  }

  // Defines the full length of the fibre
  if (depth == 2) {
    if (static_cast<int>(shortFL.size()) > ieta)
      length = shortFL[ieta] + gpar[0];
  } else {
    if (static_cast<int>(longFL.size()) > ieta)
      length = longFL[ieta];
  }
  zFibre = length;  // from beginning of abs (full length)

  if (fromEndAbs > 0) {
    zFibre -= gpar[1];  // length from end of HF
  } else {
    double zz = 0.5 * gpar[1] + point.z();  // depth of point of photon emission (from beginning of HF)
    zFibre -= zz;                           // length of fiber from point of photon emission
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::zShift for point " << point << " (R = " << hR / CLHEP::cm
                               << " cm, Index = " << ieta << ", depth = " << depth
                               << ", Fibre Length = " << length / CLHEP::cm << " cm = " << zFibre / CLHEP::cm << " cm)";
#endif
  return zFibre;
}
