#ifndef SimG4Core_PhysicsLists_CMSThermalNeutrons_h
#define SimG4Core_PhysicsLists_CMSThermalNeutrons_h

#include "G4VHadronPhysics.hh"
#include "globals.hh"

class CMSThermalNeutrons : public G4VHadronPhysics {

public: 
  CMSThermalNeutrons(G4int ver);
  ~CMSThermalNeutrons() override;

  void ConstructProcess() override;

private:
  G4int               verbose;
};

#endif






