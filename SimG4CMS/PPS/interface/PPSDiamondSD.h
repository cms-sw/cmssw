///////////////////////////////////////////////////////////////////////////////
//Author: Seyed Mohsen Etesami
// setesami@cern.ch
// 2016 Nov
///////////////////////////////////////////////////////////////////////////////

#ifndef PPS_PPSDiamondSD_h
#define PPS_PPSDiamondSD_h

#include "SimG4CMS/PPS/interface/PPSDiamondG4Hit.h"
#include "SimG4CMS/PPS/interface/PPSDiamondG4HitCollection.h"
#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include <string>

class G4Step;
class G4HCofThisEvent;
class TrackingSlaveSD;
class SimTrackManager;

class PPSDiamondSD : public SensitiveTkDetector,
                     public Observer<const BeginOfEvent*>,
                     public Observer<const EndOfEvent*> {
public:
  PPSDiamondSD(const std::string&, const SensitiveDetectorCatalog&, edm::ParameterSet const&, const SimTrackManager*);
  ~PPSDiamondSD() override;

  // Geant4 methods
  bool ProcessHits(G4Step* step, G4TouchableHistory* tHistory) override;

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void PrintAll() override;

  // CMSSW methods
  uint32_t setDetUnitId(const G4Step*) override;
  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;

protected:
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;

private:
  G4ThreeVector setToLocal(const G4ThreeVector& globalPoint);
  void stepInfo(const G4Step* aStep);
  void importInfoToHit();  //added pps
  void storeHit(PPSDiamondG4Hit*);
  void printHitInfo();

  std::unique_ptr<TrackingSlaveSD> slave_;
  std::unique_ptr<PPSVDetectorOrganization> numberingScheme_;

  PPSDiamondG4HitCollection* theHC_ = nullptr;
  PPSDiamondG4Hit* currentHit_ = nullptr;
  G4Track* theTrack_ = nullptr;
  G4VPhysicalVolume* currentPV_ = nullptr;
  G4int hcID_ = -1;
  G4int primaryID_ = 0;
  G4int parentID_ = 0;
  G4int tSliceID_ = 0;
  G4double tSlice_ = 0.0;

  G4StepPoint* preStepPoint_ = nullptr;
  G4StepPoint* postStepPoint_ = nullptr;
  G4ThreeVector hitPoint_;
  G4ThreeVector exitPoint_;
  G4ThreeVector theLocalEntryPoint_;
  G4ThreeVector theLocalExitPoint_;

  double incidentEnergy_ = 0.0;
  double pabs_ = 0.0;
  double thePx_ = 0.0;
  double thePy_ = 0.0;
  double thePz_ = 0.0;
  double tof_ = 0.0;
  double eloss_ = 0.0;

  double thetaAtEntry_ = 0.0;
  double phiAtEntry_ = 0.0;

  double vx_ = 0.0;
  double vy_ = 0.0;
  double vz_ = 0.0;

  double Globaltimehit_;
  double theglobaltimehit_;

  unsigned int unitID_ = 0;
  int verbosity_;
  int eventno_ = 0;
  short particleType_ = 0;
};

#endif  // PPS_PPSDiamondSD_h
