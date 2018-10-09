#ifndef PPSTOOLS_UTILITIES
#define PPSTOOLS_UTILITIES
#include <cmath>
#include <string>
#include <CLHEP/Units/PhysicalConstants.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>


class H_BeamParticle;
namespace CLHEP
{
     class HepLorentzVector;
}

namespace HepMC
{
    class GenParticle;
}

namespace PPSTools {

double fCrossingAngleBeam1;
double fCrossingAngleBeam2;
double fBeamMomentum;
double fBeamEnergy;
const double   urad     = 1./1000000.; 
static const double ProtonMass=CLHEP::proton_mass_c2/GeV;
static const double ProtonMassSQ=pow(ProtonMass,2);

CLHEP::HepLorentzVector HectorParticle2LorentzVector(H_BeamParticle hp,int );

H_BeamParticle LorentzVector2HectorParticle(CLHEP::HepLorentzVector p);

void LorentzBoost(H_BeamParticle& h_p,int dir, const std::string& frame);

void LorentzBoost(CLHEP::HepLorentzVector& p_out, const std::string& frame);

void LorentzBoost(HepMC::GenParticle& p_out, const std::string& frame);

void Get_t_and_xi(const CLHEP::HepLorentzVector* proton,double& t,double& xi) ;

};
#endif
