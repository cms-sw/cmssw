#ifndef Forward_PltSD_h
#define Forward_PltSD_h
 
// system include files

// user include files

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"

#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
 
#include <string>

class TrackInformation;
class SimTrackManager;
class TrackingSlaveSD;
class UpdatablePSimHit;
class G4ProcessTypeEnumerator;
class G4ParticleDefinition;

class PltSD : public SensitiveTkDetector,
  public Observer<const BeginOfEvent*>{

public:

  PltSD(const std::string&, const DDCompactView &, 
	const SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager*);
  ~PltSD() override;

  bool     ProcessHits(G4Step *,G4TouchableHistory *) override;
  uint32_t setDetUnitId(const G4Step*) override;
  void EndOfEvent(G4HCofThisEvent*) override;

  void fillHits (edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;

private:

  virtual void   sendHit();
  virtual void   updateHit(const G4Step *);
  virtual bool   newHit(const G4Step *);
  virtual bool   closeHit(const G4Step *);
  virtual void   createHit(const G4Step *);

protected:

  void           update(const BeginOfEvent *) override;

private:

  TrackingSlaveSD* slave;
  G4ProcessTypeEnumerator * theG4ProcessTypeEnumerator;
  UpdatablePSimHit * mySimHit;
  double energyCut;
  double energyHistoryCut;

  Local3DPoint globalEntryPoint;
  Local3DPoint globalExitPoint;
  const G4VPhysicalVolume * oldVolume;
  uint32_t lastId;
  int lastTrack;
  const G4ParticleDefinition* particle;
};

#endif
