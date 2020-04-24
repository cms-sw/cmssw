#ifndef SimG4Core_PhysicsList_H
#define SimG4Core_PhysicsList_H

#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepPDT/ParticleDataTable.hh"
#include "G4VModularPhysicsList.hh"

namespace sim {
  class ChordFinderSetter;
}

class PhysicsList : public G4VModularPhysicsList {

public:
  PhysicsList(G4LogicalVolumeToDDLogicalPartMap & map,
	      const HepPDT::ParticleDataTable * table_,
	      sim::ChordFinderSetter *chordFinderSetter_,
	      const edm::ParameterSet & p);
  ~PhysicsList() override;
  void SetCuts() override;

};

#endif
