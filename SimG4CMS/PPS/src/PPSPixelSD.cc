// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelSD
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: F.Ferro
//         Created:  May 4, 2015
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/PPS/interface/PPSPixelSD.h"
#include "SimG4CMS/PPS/interface/PPSPixelNumberingScheme.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

PPSPixelSD::PPSPixelSD(const std::string& name_,
                       const edm::EventSetup& es,
                       const SensitiveDetectorCatalog& clg,
                       edm::ParameterSet const& p,
                       SimTrackManager const* manager)
    : SensitiveTkDetector(name_, clg),
      numberingScheme_(nullptr),
      hcID_(-1),
      theHC_(nullptr),
      theManager_(manager),
      currentHit_(nullptr),
      theTrack_(nullptr),
      currentPV_(nullptr),
      unitID_(0),
      previousUnitID_(0),
      preStepPoint_(nullptr),
      postStepPoint_(nullptr),
      eventno_(0) {
  //Add PPS Sentitive Detector Names
  collectionName.insert(name_);

  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("PPSPixelSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  SetVerboseLevel(verbn);
  slave_ = std::make_unique<TrackingSlaveSD>(name_);
  if (name_ == "CTPPSPixelHits") {
    numberingScheme_ = std::make_unique<PPSPixelNumberingScheme>();
  } else {
    edm::LogWarning("PPSSim") << "PPSPixelSD: ReadoutName not supported\n";
  }

  edm::LogInfo("PPSSim") << "PPSPixelSD: Instantiation completed";
}

PPSPixelSD::~PPSPixelSD() {}

bool PPSPixelSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (!aStep)
    return true;

  stepInfo(aStep);
  if (!hitExists() && edeposit_ > 0.)
    createNewHit();
  return true;
}

uint32_t PPSPixelSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void PPSPixelSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("PPSSim") << "PPSPixelSD : Initialize called for " << name_;

  theHC_ = new PPSPixelG4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(name_, collectionName[0]);

  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);

  tsID_ = -2;
  primID_ = -2;
}

void PPSPixelSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries() && j < maxPixelHits_; j++) {
    PPSPixelG4Hit* aHit = (*theHC_)[j];
#ifdef debug
    LogDebug("PPSSim") << "HIT NUMERO " << j << "unit ID = " << aHit->unitID() << "\n"
                       << "               "
                       << "enrty z " << aHit->entry().z() << "\n"
                       << "               "
                       << "theta   " << aHit->thetaAtEntry() << "\n";
#endif

    Local3DPoint Enter(aHit->entryPoint().x(), aHit->entryPoint().y(), aHit->entryPoint().z());
    Local3DPoint Exit(aHit->exitPoint().x(), aHit->exitPoint().y(), aHit->exitPoint().z());
    slave_->processHits(PSimHit(Enter,
                                Exit,
                                aHit->p(),
                                aHit->tof(),
                                aHit->energyLoss(),
                                aHit->particleType(),
                                aHit->unitID(),
                                aHit->trackID(),
                                aHit->thetaAtEntry(),
                                aHit->phiAtEntry()));
  }
  summarize();
}

void PPSPixelSD::clear() {}

void PPSPixelSD::DrawAll() {}

void PPSPixelSD::PrintAll() {
  LogDebug("PPSSim") << "PPSPixelSD: Collection " << theHC_->GetName();
  theHC_->PrintAllHits();
}

void PPSPixelSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave_->name() == n) {
    c = slave_->hits();
  }
}

void PPSPixelSD::update(const BeginOfEvent* i) {
  LogDebug("PPSSim") << " Dispatched BeginOfEvent for " << GetName() << " !";
  clearHits();
  eventno_ = (*i)()->GetEventID();
}

void PPSPixelSD::update(const ::EndOfEvent*) {}

void PPSPixelSD::clearHits() { slave_->Initialize(); }

G4ThreeVector PPSPixelSD::setToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint_->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return localPoint;
}

void PPSPixelSD::stepInfo(const G4Step* aStep) {
  preStepPoint_ = aStep->GetPreStepPoint();
  postStepPoint_ = aStep->GetPostStepPoint();
  theTrack_ = aStep->GetTrack();
  Local3DPoint TheEntryPoint = SensitiveDetector::InitialStepPosition(aStep, LocalCoordinates);
  Local3DPoint TheExitPoint = SensitiveDetector::FinalStepPosition(aStep, LocalCoordinates);

#ifdef _PRINT_HITS_
  LogDebug("PPSSim") << "theEntryPoint_ " << TheEntryPoint << "\n";
  LogDebug("PPSSim") << "position " << preStepPoint_->GetPosition() << "\n";
#endif
  hitPoint_ = preStepPoint_->GetPosition();
  currentPV_ = preStepPoint_->GetPhysicalVolume();

  G4String name_ = currentPV_->GetName();
  name_.assign(name_, 0, 4);
  G4String particleType = theTrack_->GetDefinition()->GetParticleName();
  edeposit_ = aStep->GetTotalEnergyDeposit();

  tSlice_ = (postStepPoint_->GetGlobalTime()) / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);
#ifdef debug
  LogDebug("PPSSim") << "UNIT ID " << unitID_;
#endif
  primaryID_ = theTrack_->GetTrackID();

  theEntryPoint_.setX(TheEntryPoint.x());
  theEntryPoint_.setY(TheEntryPoint.y());
  theEntryPoint_.setZ(TheEntryPoint.z());
  theExitPoint_.setX(TheExitPoint.x());
  theExitPoint_.setY(TheExitPoint.y());
  theExitPoint_.setZ(TheExitPoint.z());

  position_ = hitPoint_;
  Pabs_ = aStep->GetPreStepPoint()->GetMomentum().mag() / GeV;
  Tof_ = aStep->GetPostStepPoint()->GetGlobalTime() / nanosecond;

  Eloss_ = aStep->GetTotalEnergyDeposit() / GeV;
  ParticleType_ = theTrack_->GetDefinition()->GetPDGEncoding();

  ThetaAtEntry_ = aStep->GetPreStepPoint()->GetPosition().theta();
  PhiAtEntry_ = aStep->GetPreStepPoint()->GetPosition().phi();

  ParentId_ = theTrack_->GetParentID();
  Vx_ = theTrack_->GetVertexPosition().x();
  Vy_ = theTrack_->GetVertexPosition().y();
  Vz_ = theTrack_->GetVertexPosition().z();
}

bool PPSPixelSD::hitExists() {
  if (primaryID_ < 1) {
    edm::LogWarning("PPSSim") << "***** PPSPixelSD error: primaryID = " << primaryID_ << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track
  if (tSliceID_ == tsID_ && unitID_ == previousUnitID_) {
    updateHit();
    return true;
  }

  // Reset entry point for new primary
  if (primaryID_ != primID_)
    resetForNewPrimary();

  //look in the HitContainer whether a hit with the same primID_, unitID_,
  //tSliceID_ already exists:
  bool found = false;
  int nhits = theHC_->entries();
  for (int j = 0; j < nhits && !found; j++) {
    PPSPixelG4Hit* aPreviousHit = (*theHC_)[j];
    if (aPreviousHit->trackID() == primaryID_ && aPreviousHit->timeSliceID() == tSliceID_ &&
        aPreviousHit->unitID() == unitID_) {
      currentHit_ = aPreviousHit;
      found = true;
    }
  }

  if (found) {
    updateHit();
    return true;
  }
  return false;
}

void PPSPixelSD::createNewHit() {
#ifdef debug
  LogDebug("PPSSim") << "PPSPixelSD CreateNewHit for"
                     << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo()
                     << " MVid = " << currentPV_->GetMother()->GetCopyNo() << " Unit " << unitID_ << "\n"
                     << " primary " << primaryID_ << " time slice " << tSliceID_ << " For Track  "
                     << theTrack_->GetTrackID() << " which is a " << theTrack_->GetDefinition()->GetParticleName();

  if (theTrack_->GetTrackID() == 1) {
    LogDebug("PPSSim") << " of energy " << theTrack_->GetTotalEnergy();
  } else {
    LogDebug("PPSSim") << " daughter of part. " << theTrack_->GetParentID();
  }

  if (theTrack_->GetCreatorProcess() != NULL)
    LogDebug("PPSSim") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("PPSSim") << "NO process";
#endif

  currentHit_ = new PPSPixelG4Hit;
  currentHit_->setTrackID(primaryID_);
  currentHit_->setTimeSlice(tSlice_);
  currentHit_->setUnitID(unitID_);
  currentHit_->setIncidentEnergy(incidentEnergy_);

  currentHit_->setP(Pabs_);
  currentHit_->setTof(Tof_);
  currentHit_->setEnergyLoss(Eloss_);
  currentHit_->setParticleType(ParticleType_);
  currentHit_->setThetaAtEntry(ThetaAtEntry_);
  currentHit_->setPhiAtEntry(PhiAtEntry_);

  currentHit_->setMeanPosition(position_);
  currentHit_->setEntryPoint(theEntryPoint_);
  currentHit_->setExitPoint(theExitPoint_);

  currentHit_->setParentId(ParentId_);
  currentHit_->setVx(Vx_);
  currentHit_->setVy(Vy_);
  currentHit_->setVz(Vz_);

  updateHit();

  storeHit(currentHit_);
}

void PPSPixelSD::updateHit() {
  if (Eloss_ > 0.) {
#ifdef debug
    LogDebug("PPSSim") << "G4PPSPixelSD updateHit: add eloss " << Eloss_ << "\nCurrentHit=" << currentHit_
                       << ", PostStepPoint=" << postStepPoint_->GetPosition();
#endif
    currentHit_->setEnergyLoss(Eloss_);
  }
  // buffer for next steps:
  tsID_ = tSliceID_;
  primID_ = primaryID_;
  previousUnitID_ = unitID_;
}

void PPSPixelSD::storeHit(PPSPixelG4Hit* hit) {
  if (primID_ < 0)
    return;
  if (hit == nullptr) {
    edm::LogWarning("PPSSim") << "PPSPixelSD: hit to be stored is NULL !!";
    return;
  }

  theHC_->insert(hit);
}

void PPSPixelSD::resetForNewPrimary() {
  entrancePoint_ = setToLocal(hitPoint_);

  incidentEnergy_ = preStepPoint_->GetKineticEnergy();
}

void PPSPixelSD::summarize() {}
