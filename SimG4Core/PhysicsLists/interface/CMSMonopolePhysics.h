#ifndef SimG4Core_PhysicsLists_CMSMonopolePhysics_h
#define SimG4Core_PhysicsLists_CMSMonopolePhysics_h

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4ParticleDefinition.hh"
#include "globals.hh"

#include <vector>
#include <string>

class CMSMonopolePhysics : public G4VPhysicsConstructor {

public:
  CMSMonopolePhysics(const HepPDT::ParticleDataTable * table_, G4double charge, G4int ver);
  virtual ~CMSMonopolePhysics();

  void ConstructParticle();
  void ConstructProcess();

private:
  G4int                    verbose, magCharge;
  G4double                 magn;
  std::vector<std::string> names;
  std::vector<double>      masses;
  std::vector<int>         elCharges, pdgEncodings;
  std::vector<G4ParticleDefinition*> monopoles;
};

#endif






