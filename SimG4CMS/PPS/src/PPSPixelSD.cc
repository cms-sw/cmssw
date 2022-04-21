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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/PPS/interface/PPSPixelSD.h"
#include "SimG4CMS/PPS/interface/PPSPixelOrganization.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

PPSPixelSD::PPSPixelSD(const std::string& pname,
                       const SensitiveDetectorCatalog& clg,
                       edm::ParameterSet const& p,
                       SimTrackManager const* manager)
    : SensitiveTkDetector(pname, clg), theManager_(manager) {
  //Add PPS Sentitive Detector Names
  collectionName.insert(pname);

  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("PPSPixelSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
  SetVerboseLevel(verbn);
  slave_ = std::make_unique<TrackingSlaveSD>(pname);
  if (pname == "CTPPSPixelHits") {
    numberingScheme_ = std::make_unique<PPSPixelOrganization>();
  } else {
    edm::LogWarning("PPSSim") << "PPSPixelSD: ReadoutName " << pname << " not supported";
  }
  edm::LogVerbatim("PPSSim") << "PPSPixelSD: Instantiation completed for " << pname;
}

PPSPixelSD::~PPSPixelSD() {}

bool PPSPixelSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  eloss_ = aStep->GetTotalEnergyDeposit();
  if (eloss_ > 0.0) {
    eloss_ /= GeV;
    stepInfo(aStep);
    LogDebug("PPSSim") << "PPSPixelSD: ProcessHits: edep=" << eloss_ << " "
                       << theTrack_->GetDefinition()->GetParticleName();
    if (!hitExists())
      createNewHit();
  }
  return true;
}

uint32_t PPSPixelSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void PPSPixelSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("PPSSim") << "PPSPixelSD : Initialize called for " << GetName();

  theHC_ = new PPSPixelG4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(GetName(), collectionName[0]);

  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);

  tsID_ = -2;
  edm::LogVerbatim("PPSSim") << "PPSPixelSD: is initialized " << GetName();
}

void PPSPixelSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries(); ++j) {
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
}

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
  LogDebug("PPSSim") << "PPSPixelSD: dispatched BeginOfEvent for " << GetName();
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

  hitPoint_ = preStepPoint_->GetPosition();
  exitPoint_ = postStepPoint_->GetPosition();
  currentPV_ = preStepPoint_->GetPhysicalVolume();
  theLocalEntryPoint_ = setToLocal(hitPoint_);
  theLocalExitPoint_ = setToLocal(exitPoint_);

  tSlice_ = postStepPoint_->GetGlobalTime() / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);
#ifdef debug
  LogDebug("PPSSim") << "UNIT ID " << unitID_;
#endif
  primaryID_ = theTrack_->GetTrackID();
  parentID_ = theTrack_->GetParentID();

  incidentEnergy_ = theTrack_->GetTotalEnergy() / GeV;

  pabs_ = aStep->GetPreStepPoint()->GetMomentum().mag() / GeV;
  tof_ = aStep->GetPostStepPoint()->GetGlobalTime() / nanosecond;

  particleType_ = theTrack_->GetDefinition()->GetPDGEncoding();

  thetaAtEntry_ = aStep->GetPreStepPoint()->GetPosition().theta();
  phiAtEntry_ = aStep->GetPreStepPoint()->GetPosition().phi();

  vx_ = theTrack_->GetVertexPosition().x();
  vy_ = theTrack_->GetVertexPosition().y();
  vz_ = theTrack_->GetVertexPosition().z();
}

bool PPSPixelSD::hitExists() {
  // Update if in the same detector, time-slice and for same track
  if (tSliceID_ == tsID_ && unitID_ == previousUnitID_) {
    updateHit();
    return true;
  }

  // look in the HitContainer whether a hit with the same unitID_,
  // tSliceID_ already exists: secondary energy deposition
  // is added to the primary hit
  int nhits = theHC_->entries();
  for (int j = 0; j < nhits; ++j) {
    PPSPixelG4Hit* aPreviousHit = (*theHC_)[j];
    if (aPreviousHit->timeSliceID() == tSliceID_ && aPreviousHit->unitID() == unitID_) {
      currentHit_ = aPreviousHit;
      updateHit();
      return true;
    }
  }
  return false;
}

void PPSPixelSD::createNewHit() {
#ifdef debug
  LogDebug("PPSSim") << "PPSPixelSD::CreateNewHit for"
                     << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo()
                     << " MVid = " << currentPV_->GetMother()->GetCopyNo() << " Unit " << unitID_ << "\n"
                     << " primary " << primaryID_ << " time slice " << tSliceID_ << " For Track  "
                     << " which is a " << theTrack_->GetDefinition()->GetParticleName();

  if (parentID == 0) {
    LogDebug("PPSSim") << "PPSPixelSD: primary of energy " << incidentEnergy_;
  } else {
    LogDebug("PPSSim") << " daughter of track: " << parentID;
  }

  if (theTrack_->GetCreatorProcess() != nullptr)
    LogDebug("PPSSim") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("PPSSim") << "NO process";
#endif

  currentHit_ = new PPSPixelG4Hit;
  currentHit_->setTrackID(primaryID_);
  currentHit_->setTimeSlice(tSlice_);
  currentHit_->setUnitID(unitID_);
  currentHit_->setIncidentEnergy(incidentEnergy_);

  currentHit_->setP(pabs_);
  currentHit_->setTof(tof_);
  currentHit_->setParticleType(particleType_);
  currentHit_->setThetaAtEntry(thetaAtEntry_);
  currentHit_->setPhiAtEntry(phiAtEntry_);

  G4ThreeVector pos = 0.5 * (hitPoint_ + exitPoint_);
  currentHit_->setMeanPosition(pos);
  currentHit_->setEntryPoint(theLocalEntryPoint_);
  currentHit_->setExitPoint(theLocalExitPoint_);

  currentHit_->setParentId(parentID_);
  currentHit_->setVx(vx_);
  currentHit_->setVy(vy_);
  currentHit_->setVz(vz_);

  updateHit();
  storeHit(currentHit_);
}

void PPSPixelSD::updateHit() {
#ifdef debug
  LogDebug("PPSSim") << "G4PPSPixelSD updateHit: add eloss=" << eloss_ << "\nCurrentHit=" << currentHit_
                     << ", PostStepPoint=" << postStepPoint_->GetPosition();
#endif
  currentHit_->setEnergyLoss(eloss_);
  // buffer for next steps:
  tsID_ = tSliceID_;
  previousUnitID_ = unitID_;
}

void PPSPixelSD::storeHit(PPSPixelG4Hit* hit) {
  if (hit == nullptr) {
    edm::LogWarning("PPSSim") << "PPSPixelSD: hit to be stored is NULL !!";
    return;
  }

  theHC_->insert(hit);
}
