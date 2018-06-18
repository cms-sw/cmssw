//
#ifndef Bsc_BscSD_h
#define Bsc_BscSD_h
//

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"

#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"
  
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

class BscSD : public SensitiveTkDetector,
              public Observer<const BeginOfEvent *>{

public:
  
  BscSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager* );

  ~BscSD() override;
  
  bool ProcessHits(G4Step *,G4TouchableHistory *) override;
  uint32_t  setDetUnitId(const G4Step*) override;

  void Initialize(G4HCofThisEvent * HCE) override;
  void EndOfEvent(G4HCofThisEvent * eventHC) override;
  void PrintAll() override;

  double getEnergyDeposit(const G4Step* step);
  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;
  
protected:

  void          update(const BeginOfEvent *) override;
  
private:
  G4ThreeVector setToLocal(const G4ThreeVector& global);
  G4ThreeVector setToLocalExit(const G4ThreeVector& globalPoint);
  void          getStepInfo(const G4Step* aStep);
  bool          hitExists();
  void          createNewHit();
  void          updateHit();
  void          storeHit(BscG4Hit*);
  void          resetForNewPrimary();
  
  TrackingSlaveSD* slave;
  BscNumberingScheme * numberingScheme;
  
  G4ThreeVector entrancePoint, exitPoint;
  G4ThreeVector theEntryPoint ;
  G4ThreeVector theExitPoint  ;
  
  float                incidentEnergy;
  G4int                primID  ; 

  G4int                    hcID;
  BscG4HitCollection*       theHC; 
  const SimTrackManager*      theManager;
 
  G4int                    tsID; 
  BscG4Hit*               currentHit;
  const G4Track*           theTrack;
  const G4VPhysicalVolume* currentPV;
  uint32_t             unitID, previousUnitID;
  G4int                primaryID, tSliceID;  
  G4double             tSlice;
  
  const G4StepPoint*   preStepPoint; 
  const G4StepPoint*   postStepPoint; 
  float                edeposit;
  
  G4ThreeVector        hitPoint;
  G4ThreeVector        hitPointExit;
  G4ThreeVector        hitPointLocal;
  G4ThreeVector        hitPointLocalExit;
  float Pabs;
  float Tof;
  int particleCode; 
  
  float ThetaAtEntry;
  float PhiAtEntry;
  
  int ParentId;
  float Vx,Vy,Vz;
  float X,Y,Z;
  float edepositEM, edepositHAD;
};

#endif // BscSD_h




