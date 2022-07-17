// File: TotemRPSD.cc
// Date: 18.10.2005
// Description: Sensitive Detector class for TOTEM RP Detectors
// Modifications:
#include "SimG4CMS/PPS/interface/TotemRPSD.h"
#include "SimG4CMS/PPS/interface/PPSStripOrganization.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <vector>
#include <string>

static constexpr double rp_garage_position_ = 40.0;

TotemRPSD::TotemRPSD(const std::string& pname,
                     const SensitiveDetectorCatalog& clg,
                     edm::ParameterSet const& p,
                     const SimTrackManager* manager)
    : SensitiveTkDetector(pname, clg) {
  collectionName.insert(pname);

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("TotemRPSD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");

  slave_ = std::make_unique<TrackingSlaveSD>(pname);

  if (pname == "TotemHitsRP") {
    numberingScheme_ = std::make_unique<PPSStripOrganization>();
  } else {
    edm::LogWarning("TotemRP") << "TotemRPSD: ReadoutName " << pname << " not supported";
  }

  edm::LogVerbatim("TotemRP") << "TotemRPSD: Instantiation completed for " << pname;
}

TotemRPSD::~TotemRPSD() {}

void TotemRPSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("TotemRP") << "TotemRPSD : Initialize called for " << GetName();

  theHC_ = new TotemRPG4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(GetName(), collectionName[0]);

  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);
  LogDebug("TotemRP") << "TotemRPSD: is initialized for " << GetName();
}

void TotemRPSD::printHitInfo() {
  LogDebug("TotemRP") << theTrack_->GetDefinition()->GetParticleName() << " TotemRPSD CreateNewHit for"
                      << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo() << " Unit "
                      << unitID_;
  LogDebug("TotemRP") << " primary " << primaryID_ << " time slice " << tSliceID_ << " of energy "
                      << theTrack_->GetTotalEnergy() << " Eloss_ " << eloss_ << " position pre: " << hitPoint_
                      << " post: " << exitPoint_;

  if (parentID_ == 0) {
    LogDebug("TotemRP") << " primary particle ";
  } else {
    LogDebug("TotemRP") << " daughter of part. " << parentID_;
  }

  LogDebug("TotemRP") << " and created by ";

  if (theTrack_->GetCreatorProcess() != nullptr)
    LogDebug("TotemRP") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("TotemRP") << "NO process";
}

bool TotemRPSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  eloss_ = aStep->GetTotalEnergyDeposit();
  if (eloss_ > 0.0) {
    eloss_ /= GeV;
    stepInfo(aStep);
    edm::LogVerbatim("TotemRP") << "TotemRPSD: ProcessHits 1: Eloss=" << eloss_ << " "
                                << theTrack_->GetDefinition()->GetParticleName();
    createNewHit();
  }
  return true;
}

void TotemRPSD::stepInfo(const G4Step* aStep) {
  preStepPoint_ = aStep->GetPreStepPoint();
  postStepPoint_ = aStep->GetPostStepPoint();
  theTrack_ = aStep->GetTrack();
  hitPoint_ = preStepPoint_->GetPosition();
  exitPoint_ = postStepPoint_->GetPosition();
  currentPV_ = preStepPoint_->GetPhysicalVolume();
  theLocalEntryPoint_ = setToLocal(hitPoint_);
  theLocalExitPoint_ = setToLocal(exitPoint_);

  tSlice_ = (postStepPoint_->GetGlobalTime()) / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);

  if (verbosity_)
    LogDebug("TotemRP") << "UNIT " << unitID_;

  primaryID_ = theTrack_->GetTrackID();
  parentID_ = theTrack_->GetParentID();

  incidentEnergy_ = theTrack_->GetTotalEnergy() / GeV;

  pabs_ = preStepPoint_->GetMomentum().mag() / GeV;
  thePx_ = preStepPoint_->GetMomentum().x() / GeV;
  thePy_ = preStepPoint_->GetMomentum().y() / GeV;
  thePz_ = preStepPoint_->GetMomentum().z() / GeV;

  tof_ = postStepPoint_->GetGlobalTime() / nanosecond;
  particleType_ = theTrack_->GetDefinition()->GetPDGEncoding();

  //corrected phi and theta treatment
  G4ThreeVector gmd = preStepPoint_->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd =
      ((G4TouchableHistory*)(preStepPoint_->GetTouchable()))->GetHistory()->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  thetaAtEntry_ = lnmd.theta();
  phiAtEntry_ = lnmd.phi();

  vx_ = theTrack_->GetVertexPosition().x() / mm;
  vy_ = theTrack_->GetVertexPosition().y() / mm;
  vz_ = theTrack_->GetVertexPosition().z() / mm;
}

uint32_t TotemRPSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void TotemRPSD::storeHit(TotemRPG4Hit* hit) {
  if (hit == nullptr) {
    if (verbosity_)
      LogDebug("TotemRP") << "TotemRPSD: hit to be stored is NULL !!" << std::endl;
  } else {
    theHC_->insert(hit);
  }
}

void TotemRPSD::createNewHit() {
  // Protect against creating hits in detectors not inserted
  double outrangeX = hitPoint_.x();
  double outrangeY = hitPoint_.y();
  if (std::abs(outrangeX) > rp_garage_position_ || std::abs(outrangeY) > rp_garage_position_)
    return;
  // end protection

  currentHit_ = new TotemRPG4Hit;
  currentHit_->setTrackID(primaryID_);
  currentHit_->setTimeSlice(tSlice_);
  currentHit_->setUnitID(unitID_);
  currentHit_->setIncidentEnergy(incidentEnergy_);

  currentHit_->setP(pabs_);
  currentHit_->setTof(tof_);
  currentHit_->setEnergyLoss(eloss_);
  currentHit_->setParticleType(particleType_);
  currentHit_->setThetaAtEntry(thetaAtEntry_);
  currentHit_->setPhiAtEntry(phiAtEntry_);

  currentHit_->setEntry(hitPoint_);
  currentHit_->setExit(exitPoint_);
  currentHit_->setLocalEntry(theLocalEntryPoint_);
  currentHit_->setLocalExit(theLocalExitPoint_);

  currentHit_->setParentId(parentID_);
  currentHit_->setVx(vx_);
  currentHit_->setVy(vy_);
  currentHit_->setVz(vz_);

  currentHit_->setPx(thePx_);
  currentHit_->setPy(thePy_);
  currentHit_->setPz(thePz_);

  storeHit(currentHit_);
}

G4ThreeVector TotemRPSD::setToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint_->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return localPoint;
}

void TotemRPSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries(); ++j) {
    TotemRPG4Hit* aHit = (*theHC_)[j];

    Local3DPoint entry(aHit->localEntry().x(), aHit->localEntry().y(), aHit->localEntry().z());
    Local3DPoint exit(aHit->localExit().x(), aHit->localExit().y(), aHit->localExit().z());
    slave_->processHits(PSimHit(entry,
                                exit,
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

void TotemRPSD::PrintAll() {
  LogDebug("TotemRP") << "TotemRPSD: Collection " << theHC_->GetName() << std::endl;
  theHC_->PrintAllHits();
}

void TotemRPSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave_->name() == n) {
    c = slave_->hits();
  }
}

void TotemRPSD::update(const BeginOfEvent* ptr) {
  clearHits();
  eventno_ = (*ptr)()->GetEventID();
}

void TotemRPSD::update(const ::EndOfEvent*) {}

void TotemRPSD::clearHits() { slave_->Initialize(); }
