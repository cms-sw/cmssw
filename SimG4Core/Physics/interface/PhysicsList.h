#ifndef SimG4Core_Physics_PhysicsList_H
#define SimG4Core_Physics_PhysicsList_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4VModularPhysicsList.hh"

class PhysicsList : public G4VModularPhysicsList {

public:
  PhysicsList(const edm::ParameterSet & p);
  ~PhysicsList() override;
  void SetCuts() override;

};

#endif
