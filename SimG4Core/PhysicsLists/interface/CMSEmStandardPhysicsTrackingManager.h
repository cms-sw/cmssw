#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManager_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManager_h

#include "G4VTrackingManager.hh"
#include "globals.hh"
#include "G4MscStepLimitType.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class G4eMultipleScattering;
class G4CoulombScattering;
class G4eIonisation;
class G4eBremsstrahlung;
class G4eplusAnnihilation;
class G4ElectronNuclearProcess;
class G4PositronNuclearProcess;

class G4GammaGeneralProcess;

class CMSEmStandardPhysicsTrackingManager : public G4VTrackingManager {
public:
  CMSEmStandardPhysicsTrackingManager(const edm::ParameterSet &p);
  ~CMSEmStandardPhysicsTrackingManager() override;

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

private:
  void TrackElectron(G4Track *aTrack);
  void TrackPositron(G4Track *aTrack);
  void TrackGamma(G4Track *aTrack);

  G4double fRangeFactor;
  G4double fGeomFactor;
  G4double fSafetyFactor;
  G4double fLambdaLimit;
  G4MscStepLimitType fStepLimitType;

  struct {
    G4eMultipleScattering *msc;
    G4eIonisation *ioni;
    G4eBremsstrahlung *brems;
    G4CoulombScattering *ss;
    G4ElectronNuclearProcess *nuc;
  } electron;

  struct {
    G4eMultipleScattering *msc;
    G4eIonisation *ioni;
    G4eBremsstrahlung *brems;
    G4eplusAnnihilation *annihilation;
    G4CoulombScattering *ss;
    G4PositronNuclearProcess *nuc;
  } positron;

  G4GammaGeneralProcess *gammaProc;

  static CMSEmStandardPhysicsTrackingManager *masterTrackingManager;
};

#endif
