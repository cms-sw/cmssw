#ifndef Forward_PltSD_h
#define Forward_PltSD_h
 
// system include files

// user include files

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"

#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
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
class G4TrackToParticleID;

class PltSD : public SensitiveTkDetector,
                             public Observer<const BeginOfEvent*>,
                             public Observer<const BeginOfTrack*>,
                             public Observer<const BeginOfJob*> {

public:

  PltSD(std::string, const DDCompactView &, 
		       SensitiveDetectorCatalog &, 
		       edm::ParameterSet const &, const SimTrackManager*);
  virtual ~PltSD();

  virtual bool     ProcessHits(G4Step *,G4TouchableHistory *);
  virtual uint32_t setDetUnitId(G4Step*);
  virtual void EndOfEvent(G4HCofThisEvent*);

  void fillHits (edm::PSimHitContainer&, std::string use);

private:

  virtual void   sendHit();
  virtual void   updateHit(G4Step *);
  virtual bool   newHit(G4Step *);
  virtual bool   closeHit(G4Step *);
  virtual void   createHit(G4Step *);
  void           update(const BeginOfEvent *);
  void           update(const BeginOfTrack *);
  void           update(const BeginOfJob *);
  virtual void   clearHits();
  TrackInformation* getOrCreateTrackInformation(const G4Track *);

private:

  TrackingSlaveSD*            slave;
  G4ProcessTypeEnumerator * theG4ProcessTypeEnumerator;
  G4TrackToParticleID * myG4TrackToParticleID;
  std::string        myName;
  UpdatablePSimHit * mySimHit;
  float energyCut;
  float energyHistoryCut;

  Local3DPoint globalEntryPoint;
  Local3DPoint globalExitPoint;
  G4VPhysicalVolume * oldVolume;
  uint32_t lastId;
  unsigned int lastTrack;
  int eventno;
  std::string pname;

};

#endif