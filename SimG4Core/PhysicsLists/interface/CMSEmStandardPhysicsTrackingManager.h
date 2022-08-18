#ifndef SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManager_h
#define SimG4Core_PhysicsLists_CMSEmStandardPhysicsTrackingManager_h

#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1100

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

class G4ComptonScattering;
class G4GammaConversion;
class G4PhotoElectricEffect;
class G4HadronInelasticProcess;

class CMSEmStandardPhysicsTrackingManager : public G4VTrackingManager {
public:
  CMSEmStandardPhysicsTrackingManager(const edm::ParameterSet &p);
  ~CMSEmStandardPhysicsTrackingManager();

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

  struct {
    G4PhotoElectricEffect *pe;
    G4ComptonScattering *compton;
    G4GammaConversion *conversion;
    G4HadronInelasticProcess *nuc;
  } gamma;

  static CMSEmStandardPhysicsTrackingManager *masterTrackingManager;
};

#endif

#endif
