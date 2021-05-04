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
#include "SimG4CMS/PPS/interface/PPSDiamondNumberingScheme.h"
#include "SimG4CMS/PPS/interface/PPSDiamondOrganization.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include <string>

class G4Step;
class G4HCofThisEvent;
class TrackingSlaveSD;
class SimTrackManager;

class PPSDiamondSD : public SensitiveTkDetector,
                     public Observer<const BeginOfEvent*>,
                     public Observer<const EndOfEvent*> {
public:
  PPSDiamondSD(const std::string&,
               const edm::EventSetup&,
               const SensitiveDetectorCatalog&,
               edm::ParameterSet const&,
               const SimTrackManager*);
  ~PPSDiamondSD() override;
  void printHitInfo();

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void clearTrack(G4Track* Track);
  void DrawAll() override;
  void PrintAll() override;
  void fillHits(edm::PSimHitContainer&, const std::string&) override;

private:
  static constexpr unsigned int maxDiamondHits_ = 15000;

  void clearHits() override;
  bool ProcessHits(G4Step* step, G4TouchableHistory* tHistory) override;
  uint32_t setDetUnitId(const G4Step* step) override;
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;
  void setNumberingScheme(PPSVDetectorOrganization* scheme);

  std::unique_ptr<TrackingSlaveSD> slave_;
  std::unique_ptr<PPSVDetectorOrganization> numberingScheme_;

  int verbosity_;
  int theMPDebug_;

  G4ThreeVector setToLocal(const G4ThreeVector& globalPoint);
  void stepInfo(const G4Step* aStep);
  G4bool hitExists();
  void importInfoToHit();  //added pps
  void updateHit();
  void storeHit(PPSDiamondG4Hit*);
  void resetForNewPrimary();
  void summarize();
  bool isPrimary(const G4Track* track);

  G4ThreeVector entrancePoint_;
  double incidentEnergy_;
  G4String name_;
  G4int hcID_;
  PPSDiamondG4HitCollection* theHC_;
  PPSDiamondG4Hit* currentHit_;
  G4Track* theTrack_;
  G4VPhysicalVolume* currentPV_;
  unsigned int unitID_;
  G4int primaryID_, tSliceID_;
  G4double tSlice_;

  G4StepPoint* preStepPoint_;
  G4StepPoint* postStepPoint_;
  G4ThreeVector hitPoint_;
  G4ThreeVector exitPoint_;
  G4ThreeVector theLocalEntryPoint_;
  G4ThreeVector theLocalExitPoint_;
  double Pabs_;
  double thePx_, thePy_, thePz_;
  double Tof_;
  double Eloss_;
  short ParticleType_;
  double ThetaAtEntry_;
  double PhiAtEntry_;
  int ParentId_;
  double Vx_, Vy_, Vz_;
  double Globaltimehit_;
  double theglobaltimehit_;
  int eventno_;
};

#endif  // PPS_PPSDiamondSD_h
