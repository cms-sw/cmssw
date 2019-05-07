#ifndef SimG4Core_PhysicsLists_CMSMonopolePhysics_h
#define SimG4Core_PhysicsLists_CMSMonopolePhysics_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Physics/interface/Monopole.h"

#include "G4VPhysicsConstructor.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "globals.hh"

#include <string>
#include <vector>

namespace sim {
class ChordFinderSetter;
}

class CMSMonopolePhysics : public G4VPhysicsConstructor {

public:
  CMSMonopolePhysics(const HepPDT::ParticleDataTable *table_,
                     const edm::ParameterSet &p);
  ~CMSMonopolePhysics() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4int verbose, magCharge;
  G4bool deltaRay, multiSc, transport;
  std::vector<std::string> names;
  std::vector<double> masses;
  std::vector<int> elCharges, pdgEncodings;
  std::vector<Monopole *> monopoles;
};

#endif
