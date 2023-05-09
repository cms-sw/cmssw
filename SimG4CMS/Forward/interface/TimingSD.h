#ifndef Forward_TimingSD_h
#define Forward_TimingSD_h
//
// Base sensitive class for timing detectors and sensors
//
// Created 17 June 2018 V.Ivantchenko
//

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"

#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"

#include <string>

class G4Step;
class G4StepPoint;
class G4Track;
class G4VPhysicalVolume;
class TrackingSlaveSD;
class SimTrackManager;
class G4ProcessTypeEnumerator;

//-------------------------------------------------------------------

class TimingSD : public SensitiveTkDetector, public Observer<const BeginOfEvent*> {
public:
  TimingSD(const std::string&, const SensitiveDetectorCatalog&, const SimTrackManager*);

  ~TimingSD() override;

  bool ProcessHits(G4Step*, G4TouchableHistory*) override;

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void PrintAll() override;

  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;

protected:
  void update(const BeginOfEvent*) override;

  // define time slices
  void setTimeFactor(double);

  // define MC truth thresholds
  void setCuts(double eCut, double historyCut);

  // by default accumulate hit for the same detector
  // and time slice, use primary information for fastest
  // Geant4 particle, check if for this detector new
  // hit can be merged with the existing one
  virtual bool checkHit(const G4Step*, BscG4Hit*);

  void setToLocal(const G4StepPoint* stepPoint, const G4ThreeVector& globalPoint, G4ThreeVector& localPoint);

  // accessors
  const G4ThreeVector& getLocalEntryPoint() const { return hitPointLocal; };
  const G4ThreeVector& getGlobalEntryPoint() const { return hitPoint; };

  // general method to assign track ID to be stored in hits
  virtual int getTrackID(const G4Track*);

private:
  void getStepInfo(const G4Step*);
  bool hitExists(const G4Step*);
  void createNewHit(const G4Step*);
  void updateHit();
  void storeHit(BscG4Hit*);

  TrackingSlaveSD* slave;
  G4ProcessTypeEnumerator* theEnumerator;

  const SimTrackManager* theManager;
  BscG4HitCollection* theHC;

  BscG4Hit* currentHit;
  const G4Track* theTrack;
  const G4StepPoint* preStepPoint;
  const G4StepPoint* postStepPoint;

  uint32_t unitID, previousUnitID;

  int primID;
  int hcID;
  int tsID;
  int primaryID;
  int tSliceID;

  G4ThreeVector hitPoint;
  G4ThreeVector hitPointExit;
  G4ThreeVector hitPointLocal;
  G4ThreeVector hitPointLocalExit;

  double tSlice;
  double timeFactor;

  double energyCut;         // MeV
  double energyHistoryCut;  // MeV

  double incidentEnergy;  // MeV
  float tof;              // ns
  float edeposit;
  float edepositEM, edepositHAD;
};

#endif
