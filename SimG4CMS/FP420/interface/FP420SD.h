//
#ifndef FP420_FP420SD_h
#define FP420_FP420SD_h
//

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

#include <string>

class TrackingSlaveSD;
//AZ:
class FrameRotation;
class TrackInformation;
class SimTrackManager;
class TrackingSlaveSD;
class UpdatablePSimHit;
class G4ProcessTypeEnumerator;
class G4TrackToParticleID;

//-------------------------------------------------------------------

class FP420SD : public SensitiveTkDetector,
                public Observer<const BeginOfRun*>,
                public Observer<const BeginOfEvent*>,
                public Observer<const EndOfEvent*> {
public:
  FP420SD(const std::string&,
          const edm::EventSetup&,
          const SensitiveDetectorCatalog&,
          edm::ParameterSet const&,
          const SimTrackManager*);

  ~FP420SD() override;

  bool ProcessHits(G4Step*, G4TouchableHistory*) override;
  uint32_t setDetUnitId(const G4Step*) override;

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  virtual double getEnergyDeposit(G4Step* step);
  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;

private:
  void update(const BeginOfRun*) override;
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;

  G4ThreeVector SetToLocal(const G4ThreeVector& global);
  G4ThreeVector SetToLocalExit(const G4ThreeVector& globalPoint);
  void GetStepInfo(G4Step* aStep);
  G4bool HitExists();
  void CreateNewHit();
  void UpdateHit();
  void StoreHit(FP420G4Hit*);
  void ResetForNewPrimary();
  void Summarize();

private:
  //AZ:
  TrackingSlaveSD* slave;
  FP420NumberingScheme* numberingScheme;

  G4ThreeVector entrancePoint, exitPoint;
  G4ThreeVector theEntryPoint;
  G4ThreeVector theExitPoint;

  float incidentEnergy;

  G4int hcID;
  FP420G4HitCollection* theHC;
  const SimTrackManager* theManager;

  G4int tsID;
  FP420G4Hit* currentHit;
  G4Track* theTrack;
  G4VPhysicalVolume* currentPV;
  // unsigned int         unitID, previousUnitID;
  uint32_t unitID, previousUnitID;
  G4int tSliceID;
  unsigned int primaryID, primID;

  G4double tSlice;

  G4StepPoint* preStepPoint;
  G4StepPoint* postStepPoint;
  float edeposit;

  G4ThreeVector hitPoint;
  G4ThreeVector hitPointExit;
  G4ThreeVector hitPointLocal;
  G4ThreeVector hitPointLocalExit;
  float Pabs;
  float Tof;
  float Eloss;
  short ParticleType;

  float ThetaAtEntry;
  float PhiAtEntry;

  int ParentId;
  float Vx, Vy, Vz;
  float X, Y, Z;

  //
  // Hist
  //
  int eventno;

protected:
  float edepositEM, edepositHAD;
  G4int emPDG;
  G4int epPDG;
  G4int gammaPDG;
};

#endif  // FP420SD_h
