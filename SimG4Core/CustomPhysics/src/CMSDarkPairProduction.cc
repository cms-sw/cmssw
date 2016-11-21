//
// Authors of this file:        Dustin Stolp (dostolp@ucdavis.edu)
//                              Sushil S. Chauhan (schauhan@cern.ch) 
// Creation date: 01.22.2015
//
// -------------------------------------------------------------------
//
#include "SimG4Core/CustomPhysics/interface/CMSDarkPairProduction.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4PairProductionRelModel.hh"

using namespace std;

static const G4double xsfactor =
  4*fine_structure_const*classic_electr_radius*classic_electr_radius;

static G4double dark_factor;

CMSDarkPairProduction::CMSDarkPairProduction(const G4ParticleDefinition* p,G4double df,const G4String& nam) : G4PairProductionRelModel(p,nam){
  dark_factor = df;

}

CMSDarkPairProduction::~CMSDarkPairProduction(){}

G4double CMSDarkPairProduction::ComputeCrossSectionPerAtom(const G4ParticleDefinition*,
                                                     G4double gammaEnergy, G4double Z,
                                                     G4double, G4double, G4double)
{
  
  G4double crossSection = 0.0 ;
  if ( gammaEnergy <= 2.0*electron_mass_c2 ) return crossSection;
  
  SetCurrentElement(Z);
  // choose calculator according to parameters and switches
  // in the moment only one calculator:
  crossSection=ComputeXSectionPerAtom(gammaEnergy,Z);
  
  G4double xi = Finel/(Fel - fCoulomb); // inelastic contribution
  crossSection *= dark_factor * xsfactor*Z*(Z+xi);
  return crossSection;
}
void
CMSDarkPairProduction::SampleSecondaries(std::vector<G4DynamicParticle*>* fvect,
               const G4MaterialCutsCouple* couple,
               const G4DynamicParticle* aDynamicGamma,
               G4double e1,
               G4double e2)
{
G4PairProductionRelModel::SampleSecondaries(fvect, couple, aDynamicGamma, e1, e2);
        
}

