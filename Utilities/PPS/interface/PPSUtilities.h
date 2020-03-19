#ifndef PPSTOOLS_UTILITIES
#define PPSTOOLS_UTILITIES
#include <cmath>
#include <string>
#include <CLHEP/Units/PhysicalConstants.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include "TLorentzVector.h"

class H_BeamParticle;

namespace HepMC {
  class GenParticle;
}

namespace PPSTools {

  struct FullBeamInfo {
    double fCrossingAngleBeam1;
    double fCrossingAngleBeam2;
    double fBeamMomentum;
    double fBeamEnergy;
  };

  struct LimitedBeamInfo {
    double fBeamMomentum;
    double fBeamEnergy;
  };

  const double urad = 1. / 1000000.;
  const double ProtonMass = CLHEP::proton_mass_c2 / GeV;
  const double ProtonMassSQ = pow(ProtonMass, 2);

  TLorentzVector HectorParticle2LorentzVector(H_BeamParticle hp, int);

  H_BeamParticle LorentzVector2HectorParticle(TLorentzVector p);

  void LorentzBoost(H_BeamParticle& h_p, int dir, const std::string& frame, FullBeamInfo const& bi);

  void LorentzBoost(TLorentzVector& p_out, const std::string& frame, FullBeamInfo const& bi);

  void LorentzBoost(HepMC::GenParticle& p_out, const std::string& frame, FullBeamInfo const& bi);

  void Get_t_and_xi(const TLorentzVector* proton, double& t, double& xi, LimitedBeamInfo const& bi);

};  // namespace PPSTools
#endif
