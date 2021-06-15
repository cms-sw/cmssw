//--------------------------------------------------------------------
//
// 10.06.2018 V.Ivanchenko EM physics of CMS mirgrated to Geant4 10.6
//                         based on option1 Geant4 EM and called EMM
// 15.04.2021              Become legacy and will retire soon
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsLPM_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsLPM : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsLPM(G4int ver);
  ~CMSEmStandardPhysicsLPM() override;

  void ConstructParticle() override;
  void ConstructProcess() override;
};

#endif
