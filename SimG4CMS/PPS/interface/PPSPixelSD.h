#ifndef _PPSPixelSD_h
#define _PPSPixelSD_h
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelSD
//
/**\class PPSPixelSD PPSPixelSD.h SimG4CMS/PPS/interface/PPSPixelSD.h
 
 Description: Stores hits of PPSPixel in appropriate  container
 
 Usage:
    Used in sensitive detector builder 
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4CMS/PPS/interface/PPSPixelG4Hit.h"
#include "SimG4CMS/PPS/interface/PPSPixelG4HitCollection.h"
#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"

#include <string>

class TrackingSlaveSD;
class SimTrackManager;
class G4Step;
class G4StepPoint;
class G4Track;

class PPSPixelSD : public SensitiveTkDetector,
                   public Observer<const BeginOfEvent*>,
                   public Observer<const EndOfEvent*> {
public:
  PPSPixelSD(const std::string&,
             const edm::EventSetup&,
             const SensitiveDetectorCatalog&,
             edm::ParameterSet const&,
             SimTrackManager const*);
  ~PPSPixelSD() override;

  bool ProcessHits(G4Step*, G4TouchableHistory*) override;
  uint32_t setDetUnitId(const G4Step*) override;

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void fillHits(edm::PSimHitContainer&, const std::string&) override;

private:
  void update(const BeginOfEvent*) override;
  void update(const ::EndOfEvent*) override;
  void clearHits() override;

private:
  static constexpr unsigned int maxPixelHits_ = 15000;
  G4ThreeVector setToLocal(const G4ThreeVector& globalPoint);
  void stepInfo(const G4Step* aStep);
  bool hitExists();
  void createNewHit();
  void updateHit();
  void storeHit(PPSPixelG4Hit*);
  void resetForNewPrimary();
  void summarize();

private:
  std::unique_ptr<TrackingSlaveSD> slave_;
  std::unique_ptr<PPSVDetectorOrganization> numberingScheme_;

  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (cristal/fiber/scintillator) and the Time slice ID.

  G4ThreeVector entrancePoint_;
  float incidentEnergy_;
  G4int primID_;  //@@ ID of the primary particle.

  std::string name_;
  G4int hcID_;
  PPSPixelG4HitCollection* theHC_;
  const SimTrackManager* theManager_;

  int tsID_;
  PPSPixelG4Hit* currentHit_;
  G4Track* theTrack_;
  G4VPhysicalVolume* currentPV_;
  uint32_t unitID_, previousUnitID_;
  int primaryID_, tSliceID_;
  double tSlice_;

  G4StepPoint* preStepPoint_;
  G4StepPoint* postStepPoint_;
  float edeposit_;
  G4ThreeVector hitPoint_;

  G4ThreeVector position_;
  G4ThreeVector theEntryPoint_;
  G4ThreeVector theExitPoint_;
  float Pabs_;
  float Tof_;
  float Eloss_;
  short ParticleType_;

  float ThetaAtEntry_;
  float PhiAtEntry_;

  int ParentId_;
  float Vx_, Vy_, Vz_;

  int eventno_;
};

#endif
