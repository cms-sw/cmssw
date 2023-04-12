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
HFFibre::Params::Params(double iFractionOfSpeedOfLightInFibre,
                        const HcalDDDSimConstants* hcons,
                        const HcalSimulationParameters* hps)
    : fractionOfSpeedOfLightInFibre_{iFractionOfSpeedOfLightInFibre},
      gParHF_{hcons->getGparHF()},
      rTableHF_{hcons->getRTableHF()},
      shortFibreLength_{hps->shortFiberLength_},
      longFibreLength_{hps->longFiberLength_},
      attenuationLength_{hps->attenuationLength_},
      lambdaLimits_{{static_cast<double>(hps->lambdaLimits_[0]), static_cast<double>(hps->lambdaLimits_[1])}} {}

HFFibre::HFFibre(const HcalDDDSimConstants* hcons, const HcalSimulationParameters* hps, edm::ParameterSet const& p)
    : HFFibre(Params(p.getParameter<edm::ParameterSet>("HFShower")
                         .getParameter<edm::ParameterSet>("HFShowerBlock")
                         .getParameter<double>("CFibre"),
                     hcons,
                     hps)) {}

HFFibre::HFFibre(Params iP)
    : cFibre_(c_light * iP.fractionOfSpeedOfLightInFibre_),
      gpar_(std::move(iP.gParHF_)),
      radius_(std::move(iP.rTableHF_)),
      shortFL_(std::move(iP.shortFibreLength_)),
      longFL_(std::move(iP.longFibreLength_)),
      attL_(std::move(iP.attenuationLength_)),
      lambLim_(iP.lambdaLimits_) {
  edm::LogVerbatim("HFShower") << "HFFibre:: Speed of light in fibre " << cFibre_ / (CLHEP::m / CLHEP::ns) << " m/ns";
  // Attenuation length
  nBinAtt_ = static_cast<int>(attL_.size());

  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFFibre: " << nBinAtt_ << " attL(1/cm): ";
    for (int it = 0; it < nBinAtt_; it++) {
      if (it / 10 * 10 == it) {
        logger << "\n";
      }
      logger << "  " << attL_[it] * CLHEP::cm;
    }
  });
  edm::LogVerbatim("HFShower") << "HFFibre: Limits on lambda " << lambLim_[0] << " and " << lambLim_[1];
  // Fibre Lengths
  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFFibre: " << longFL_.size() << " Long Fibre Length(cm):";
    for (unsigned int it = 0; it < longFL_.size(); it++) {
      if (it / 10 * 10 == it) {
        logger << "\n";
      }
      logger << "  " << longFL_[it] / CLHEP::cm;
    }
  });
  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFFibre: " << shortFL_.size() << " Short Fibre Length(cm):";
    for (unsigned int it = 0; it < shortFL_.size(); it++) {
      if (it / 10 * 10 == it) {
        logger << "\n";
      }
      logger << "  " << shortFL_[it] / CLHEP::cm;
    }
  });
  // Now geometry parameters

  nBinR_ = static_cast<int>(radius_.size());
  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFFibre: " << radius_.size() << " rTable(cm):";
    for (int i = 0; i < nBinR_; ++i) {
      if (i / 10 * 10 == i) {
        logger << "\n";
      }
      logger << "  " << radius_[i] / CLHEP::cm;
    }
  });

  edm::LogVerbatim("HFShower").log([&](auto& logger) {
    logger << "HFFibre: " << gpar_.size() << " gParHF:";
    for (std::size_t i = 0; i < gpar_.size(); ++i) {
      if (i / 10 * 10 == i) {
        logger << "\n";
      }
      logger << "  " << gpar_[i];
    }
  });
}

double HFFibre::attLength(double lambda) const {
  int i = int(nBinAtt_ * (lambda - lambLim_[0]) / (lambLim_[1] - lambLim_[0]));

  int j = i;
  if (i >= nBinAtt_)
    j = nBinAtt_ - 1;
  else if (i < 0)
    j = 0;
  double att = attL_[j];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::attLength for Lambda " << lambda << " index " << i << " " << j
                               << " Att. Length " << att;
#endif
  return att;
}

double HFFibre::tShift(const G4ThreeVector& point, int depth, int fromEndAbs) const {
  double zFibre = zShift(point, depth, fromEndAbs);
  double time = zFibre / cFibre_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::tShift for point " << point << " ( depth = " << depth
                               << ", traversed length = " << zFibre / CLHEP::cm << " cm) = " << time / CLHEP::ns
                               << " ns";
#endif
  return time;
}

double HFFibre::zShift(const G4ThreeVector& point, int depth, int fromEndAbs) const {  // point is z-local

  double zFibre = 0;
  int ieta = 0;
  double length = 250 * CLHEP::cm;
  double hR = std::sqrt((point.x()) * (point.x()) + (point.y()) * (point.y()));

  // Defines the Radius bin by radial subdivision
  if (fromEndAbs >= 0) {
    for (int i = nBinR_ - 1; i > 0; --i)
      if (hR < radius_[i])
        ieta = nBinR_ - i - 1;
  }

  // Defines the full length of the fibre
  if (depth == 2) {
    if (static_cast<int>(shortFL_.size()) > ieta)
      length = shortFL_[ieta] + gpar_[0];
  } else {
    if (static_cast<int>(longFL_.size()) > ieta)
      length = longFL_[ieta];
  }
  zFibre = length;  // from beginning of abs (full length)

  if (fromEndAbs > 0) {
    zFibre -= gpar_[1];  // length from end of HF
  } else {
    double zz = 0.5 * gpar_[1] + point.z();  // depth of point of photon emission (from beginning of HF)
    zFibre -= zz;                            // length of fiber from point of photon emission
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HFShower") << "HFFibre::zShift for point " << point << " (R = " << hR / CLHEP::cm
                               << " cm, Index = " << ieta << ", depth = " << depth
                               << ", Fibre Length = " << length / CLHEP::cm << " cm = " << zFibre / CLHEP::cm << " cm)";
#endif
  return zFibre;
}
