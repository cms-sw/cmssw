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

  G4ThreeVector entrancePoint;
  double incidentEnergy;

  G4String name;
  G4int hcID;
  TotemRPG4HitCollection* theHC;

  TotemRPG4Hit* currentHit;
  G4Track* theTrack;
  G4VPhysicalVolume* currentPV;
  unsigned int unitID;
  G4int primaryID, tSliceID;
  G4double tSlice;

  G4StepPoint* preStepPoint;
  G4StepPoint* postStepPoint;
  G4ThreeVector hitPoint;
  G4ThreeVector exitPoint;
  G4ThreeVector theLocalEntryPoint;
  G4ThreeVector theLocalExitPoint;

  double Pabs;
  double p_x, p_y, p_z;
  double Tof;
  double Eloss;
  short ParticleType;

  double ThetaAtEntry;
  double PhiAtEntry;

  int ParentId;
  double Vx, Vy, Vz;

  // Hist
  static TotemTestHitHBNtuple* theNtuple;
  int eventno;
};

#endif  // PPS_TotemRPSD_h
