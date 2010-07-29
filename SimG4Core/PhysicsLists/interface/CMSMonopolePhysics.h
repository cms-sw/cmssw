#ifndef SimG4Core_PhysicsLists_CMSMonopolePhysics_h
#define SimG4Core_PhysicsLists_CMSMonopolePhysics_h

#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/Physics/interface/G4Monopole.hh"

#include "HepPDT/ParticleDataTable.hh"
#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

#include <vector>
#include <string>

class CMSMonopolePhysics : public G4VPhysicsConstructor {

public:
  CMSMonopolePhysics(const HepPDT::ParticleDataTable * table_, sim::FieldBuilder * fB_, G4double charge, G4int ver);
  virtual ~CMSMonopolePhysics();

  void ConstructParticle();
  void ConstructProcess();

private:
  sim::FieldBuilder  *     fieldBuilder;
  G4int                    verbose, magCharge;
  std::vector<std::string> names;
  std::vector<double>      masses;
  std::vector<int>         elCharges, pdgEncodings;
  std::vector<G4Monopole*> monopoles;
};

#endif






