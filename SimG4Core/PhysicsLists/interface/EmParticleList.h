#ifndef SimG4Core_PhysicsList_EmParticleList_h
#define SimG4Core_PhysicsList_EmParticleList_h 1

// V.Ivanchenko 6 March 2017
// List of Geant4 basic particle names used in SIM step 

#include "globals.hh"
#include <vector>

class EmParticleList {

public:

  EmParticleList() {};
  ~EmParticleList() {};

  const std::vector<G4String>& PartNames();

private:
  static const std::vector<G4String>  pNames; 
  
};

#endif

