#ifndef PPS_TotemRPSD_h
#define PPS_TotemRPSD_h

/**
   In this example the TotemSensitiveDetector serves as a master for two different ExampleSD
 */

#include "SimG4CMS/PPS/interface/TotemRPG4Hit.h"
#include "SimG4CMS/PPS/interface/TotemRPG4HitCollection.h"
#include "SimG4CMS/PPS/interface/TotemRPVDetectorOrganization.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include <string>

class G4Step;
class G4HCofThisEvent;
class G4Track;
class G4StepPoint;
class TrackingSlaveSD;
class SimTrackManager;
class TotemTestHitHBNtuple;

class TotemRPSD : public SensitiveTkDetector, public Observer<const BeginOfEvent*>, public Observer<const EndOfEvent*> {
public:
  TotemRPSD(const std::string&,
            const edm::EventSetup&,
            const SensitiveDetectorCatalog&,
            edm::ParameterSet const&,
            const SimTrackManager*);
  ~TotemRPSD() override;

  void printHitInfo();

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  static constexpr double rp_garage_position_ = 40.0;

private:
  static constexpr unsigned int maxTotemHits_ = 15000;
  void clearHits() override;
  bool ProcessHits(G4Step* step, G4TouchableHistory* tHistory) override;
  uint32_t setDetUnitId(const G4Step* step) override;
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;

  void setNumberingScheme(TotemRPVDetectorOrganization* scheme);

  std::unique_ptr<TrackingSlaveSD> slave_;
  std::unique_ptr<TotemRPVDetectorOrganization> numberingScheme_;

private:
  int verbosity_;

  G4ThreeVector setToLocal(const G4ThreeVector& globalPoint);
  void stepInfo(const G4Step* aStep);
  G4bool hitExists();
  void createNewHit();
  void updateHit();
  void storeHit(TotemRPG4Hit*);
  void resetForNewPrimary();
  void summarize();
  bool isPrimary(const G4Track* track);

private:
  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (cristal/fiber/scintillator) and the Time slice ID.

  G4ThreeVector entrancePoint_;
  double incidentEnergy_;

  G4String name_;
  G4int hcID_;
  TotemRPG4HitCollection* theHC_;

  TotemRPG4Hit* currentHit_;
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

  // Hist
  //static TotemTestHitHBNtuple* theNtuple_;
  int eventno_;
};

#endif  // PPS_TotemRPSD_h
