/** fake sensitive detector modelled after PSimHit for a user example;
    the ORCA/CommonDet/BasicDet/interface/PSimHit.h and
    ORCA/CommonDet/PBasicDet/interface/PSimHitROUFactory.h
    are copied under Mantis/USDHitExample/test/stubs;
    the user must provide links to ORCA libraries PBasicDet, BasicDet and
    DetGeometry which must be loaded before the USDHitExample library -
    see sens.macro under Mantis/G4Notification/test
 */

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

  void Print_Hit_Info();

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  static constexpr double rp_garage_position_ = 40.0;

private:
  void clearHits() override;
  bool ProcessHits(G4Step* step, G4TouchableHistory* tHistory) override;
  uint32_t setDetUnitId(const G4Step* step) override;
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;

  void SetNumberingScheme(TotemRPVDetectorOrganization* scheme);

  TrackingSlaveSD* slave;
  TotemRPVDetectorOrganization* numberingScheme;

private:
  int verbosity_;

  G4ThreeVector SetToLocal(const G4ThreeVector& globalPoint);
  void GetStepInfo(const G4Step* aStep);
  G4bool HitExists();
  void CreateNewHit();
  void UpdateHit();
  void StoreHit(TotemRPG4Hit*);
  void ResetForNewPrimary();
  void Summarize();
  bool IsPrimary(const G4Track* track);

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
  static TotemTestHitHBNtuple* theNtuple_;
  int eventno_;
};

#endif  // PPS_TotemRPSD_h
