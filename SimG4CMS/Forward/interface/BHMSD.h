#ifndef SimG4CMSForward_BHMSD_h
#define SimG4CMSForward_BHMSD_h

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"

  
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

#include <string>

class TrackingSlaveSD;

class TrackInformation;
class SimTrackManager;
class TrackingSlaveSD;
class UpdatablePSimHit;
class G4ProcessTypeEnumerator;
class G4TrackToParticleID;


//-------------------------------------------------------------------

class BHMSD : public SensitiveTkDetector,
              public Observer<const BeginOfRun *>,
              public Observer<const BeginOfEvent*>,
              public Observer<const EndOfEvent*> {

public:
  
  BHMSD(std::string, const DDCompactView &, SensitiveDetectorCatalog &, 
	edm::ParameterSet const &, const SimTrackManager* );


  virtual ~BHMSD();
  
  virtual bool ProcessHits(G4Step *,G4TouchableHistory *);
  virtual uint32_t  setDetUnitId(G4Step*);

  virtual void Initialize(G4HCofThisEvent * HCE);
  virtual void EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void clear();
  virtual void DrawAll();
  virtual void PrintAll();

  virtual double getEnergyDeposit(G4Step* step);
  void           fillHits(edm::PSimHitContainer&, std::string use);
  
  std::vector<std::string> getNames();
  
private:
  void           update(const BeginOfRun *);
  void           update(const BeginOfEvent *);
  void           update(const ::EndOfEvent *);
  virtual void   clearHits();

private:
  
  G4ThreeVector SetToLocal(G4ThreeVector global);
  G4ThreeVector SetToLocalExit(G4ThreeVector globalPoint);
  void          GetStepInfo(G4Step* aStep);
  G4bool        HitExists();
  void          CreateNewHit();
  void          UpdateHit();
  void          StoreHit(BscG4Hit*);
  void          ResetForNewPrimary();
  void          Summarize();
  
  
private:
  
  TrackingSlaveSD       *slave;
  BHMNumberingScheme    *numberingScheme;
  
  G4ThreeVector          entrancePoint, exitPoint;
  G4ThreeVector          theEntryPoint, theExitPoint;
  
  float                  incidentEnergy;
  G4int                  primID  ; 
  
  std::string            name;
  G4int                  hcID;
  BscG4HitCollection*    theHC; 
  const SimTrackManager* theManager;
 
  G4int                  tsID; 
  BscG4Hit*              currentHit;
  G4Track*               theTrack;
  G4VPhysicalVolume*     currentPV;
  uint32_t               unitID, previousUnitID;
  G4int                  primaryID, tSliceID;  
  G4double               tSlice;
  
  G4StepPoint*           preStepPoint; 
  G4StepPoint*           postStepPoint; 
  float                  edeposit;
  
  G4ThreeVector          hitPoint;
  G4ThreeVector          hitPointExit;
  G4ThreeVector          hitPointLocal;
  G4ThreeVector          hitPointLocalExit;

  float                  Pabs, Tof, Eloss;	
  short                  ParticleType; 
  float                  ThetaAtEntry, PhiAtEntry;
  
  int                    ParentId;
  float                  Vx,Vy,Vz;
  float                  X,Y,Z;
  
  int                    eventno;
  
protected:
  
  float                  edepositEM, edepositHAD;
  G4int                  emPDG;
  G4int                  epPDG;
  G4int                  gammaPDG;
};

#endif




