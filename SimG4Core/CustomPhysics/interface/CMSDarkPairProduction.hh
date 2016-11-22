//
// File name:     CMSDarkPairProduction
//
// Author:        Dustin Stolp (dostolp@ucdavis.edu)
//                Sushil S. Chauhan (schauhan@cern.ch)  
// Creation date: 01.22.2015
// -------------------------------------------------------------------
//
#ifndef SimG4Core_CustomPhysics_CMSDarkPairProduction_h
#define SimG4Core_CustomPhysics_CMSDarkPairProduction_h

#include <CLHEP/Units/PhysicalConstants.h>

#include "G4PairProductionRelModel.hh"
#include "G4PhysicsTable.hh"
#include "G4NistManager.hh"
#include "G4VEmModel.hh"

class CMSDarkPairProduction : public G4PairProductionRelModel
{
public:
  CMSDarkPairProduction(const G4ParticleDefinition* p = 0,
		      const G4double df = 1E0,
                      const G4String& nam = "BetheHeitlerLPM");

  virtual ~CMSDarkPairProduction();

  virtual G4double ComputeCrossSectionPerAtom(
                      const G4ParticleDefinition*,
                      G4double kinEnergy,
                      G4double Z,
                      G4double A=0.,
                      G4double cut=0.,
                      G4double emax=DBL_MAX);

  void SampleSecondaries(std::vector<G4DynamicParticle*>* fvect,
               	      const G4MaterialCutsCouple* couple,
               	      const G4DynamicParticle* aDynamicGamma,
                      G4double e1,
                      G4double e2); 

};
#endif
