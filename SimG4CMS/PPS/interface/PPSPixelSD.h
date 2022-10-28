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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

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
  PPSPixelSD(const std::string&, const SensitiveDetectorCatalog&, edm::ParameterSet const&, SimTrackManager const*);
  ~PPSPixelSD() override;

  // Geant4 methods
  bool ProcessHits(G4Step*, G4TouchableHistory*) override;

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
  bool hitExists();
  void createNewHit();
  void updateHit();
  void storeHit(PPSPixelG4Hit*);

  std::unique_ptr<TrackingSlaveSD> slave_;
  std::unique_ptr<PPSVDetectorOrganization> numberingScheme_;

  PPSPixelG4HitCollection* theHC_ = nullptr;
  const SimTrackManager* theManager_;
  PPSPixelG4Hit* currentHit_ = nullptr;
  G4Track* theTrack_ = nullptr;
  G4VPhysicalVolume* currentPV_ = nullptr;
  G4int hcID_ = -1;

  G4StepPoint* preStepPoint_ = nullptr;
  G4StepPoint* postStepPoint_ = nullptr;
  G4ThreeVector hitPoint_;
  G4ThreeVector exitPoint_;
  G4ThreeVector theLocalEntryPoint_;
  G4ThreeVector theLocalExitPoint_;

  double tSlice_ = 0.0;
  double eloss_ = 0.0;
  float incidentEnergy_ = 0.f;
  float pabs_ = 0.f;
  float tof_ = 0.f;

  float thetaAtEntry_ = 0.f;
  float phiAtEntry_ = 0.f;
  float vx_ = 0.f;
  float vy_ = 0.f;
  float vz_ = 0.f;

  uint32_t unitID_ = 0;
  uint32_t previousUnitID_ = 0;
  int tsID_ = 0;
  int primaryID_ = 0;
  int parentID_ = 0;
  int tSliceID_ = 0;
  int eventno_ = 0;
  short particleType_ = 0;
};

#endif
