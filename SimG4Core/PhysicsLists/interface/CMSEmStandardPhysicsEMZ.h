
//--------------------------------------------------------------------
//
// 15.04.2021 V.Ivanchenko EM physics of CMS mirgrated to Geant4 10.7
//                         based on option4 Geant4 EM and called EMZ
//
//--------------------------------------------------------------------

#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMZ_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsEMZ_h

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

class CMSEmStandardPhysicsEMZ : public G4VPhysicsConstructor {
public:
  CMSEmStandardPhysicsEMZ(G4int ver);
  ~CMSEmStandardPhysicsEMZ() override;

  void ConstructParticle() override;
  void ConstructProcess() override;
};

#endif
