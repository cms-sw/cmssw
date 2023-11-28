#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMMT_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMMT_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CMSEmStandardPhysicsEMMT : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsEMMT(G4int ver, const edm::ParameterSet& p);
  ~CMSEmStandardPhysicsEMMT() override;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  const edm::ParameterSet& fParameterSet;
};

#endif
