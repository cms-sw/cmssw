//--------------------------------------------------------------------
//
// 15.04.2021 V.Ivanchenko EM physics of CMS mirgrated to Geant4 10.7
//                         based on option1 Geant4 EM and called EMM
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysics_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysics_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CMSEmStandardPhysics : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysics(G4int ver, const edm::ParameterSet& p);
  ~CMSEmStandardPhysics() override = default;

  void ConstructParticle() override;
  void ConstructProcess() override;

private:
  G4double fRangeFactor;
  G4double fGeomFactor;
  G4double fSafetyFactor;
  G4double fLambdaLimit;
  G4MscStepLimitType fStepLimitType;
};

#endif
