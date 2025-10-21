//--------------------------------------------------------------------
//
// 18.10.2025 S.Diederichs EM physics using AdePT based on
//                         CMSEmStandardPhysicsof
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsA_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsA_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class AdePTConfiguration;

class CMSEmStandardPhysicsA : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsA(G4int ver, const edm::ParameterSet& p);
  ~CMSEmStandardPhysicsA() override = default;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4double fRangeFactor;
  G4double fGeomFactor;
  G4double fSafetyFactor;
  G4double fLambdaLimit;
  G4MscStepLimitType fStepLimitType;
  AdePTConfiguration* fAdePTConfiguration;
};

#endif
