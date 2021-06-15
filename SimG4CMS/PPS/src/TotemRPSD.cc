// File: TotemRPSD.cc
// Date: 18.10.2005
// Description: Sensitive Detector class for TOTEM RP Detectors
// Modifications:
#include "SimG4CMS/PPS/interface/TotemRPSD.h"
#include "SimG4CMS/PPS/interface/PPSStripNumberingScheme.h"

#include "FWCore/Framework/interface/EventSetup.h"
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

TotemRPSD::TotemRPSD(const std::string& name_,
                     const edm::EventSetup& es,
                     const SensitiveDetectorCatalog& clg,
                     edm::ParameterSet const& p,
                     const SimTrackManager* manager)
    : SensitiveTkDetector(name_, clg),
      numberingScheme_(nullptr),
      hcID_(-1),
      theHC_(nullptr),
      currentHit_(nullptr),
      theTrack_(nullptr),
      currentPV_(nullptr),
      unitID_(0),
      preStepPoint_(nullptr),
      postStepPoint_(nullptr),
      eventno_(0) {
  collectionName.insert(name_);

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("TotemRPSD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");

  slave_ = std::make_unique<TrackingSlaveSD>(name_);

  if (name_ == "TotemHitsRP") {
    numberingScheme_ = std::make_unique<PPSStripNumberingScheme>(3);
  } else {
    edm::LogWarning("TotemRP") << "TotemRPSD: ReadoutName not supported\n";
  }

  edm::LogInfo("TotemRP") << "TotemRPSD: Instantiation completed";
}

TotemRPSD::~TotemRPSD() {}

void TotemRPSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("TotemRP") << "TotemRPSD : Initialize called for " << name_;

  theHC_ = new TotemRPG4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(name_, collectionName[0]);

  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);
}

void TotemRPSD::printHitInfo() {
  LogDebug("TotemRP") << theTrack_->GetDefinition()->GetParticleName() << " TotemRPSD CreateNewHit for"
                      << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo() << " Unit "
                      << unitID_;
  LogDebug("TotemRP") << " primary " << primaryID_ << " time slice " << tSliceID_ << " of energy "
                      << theTrack_->GetTotalEnergy() << " Eloss_ " << Eloss_ << " positions ";
  printf("(%10f,%10f,%10f)",
         preStepPoint_->GetPosition().x(),
         preStepPoint_->GetPosition().y(),
         preStepPoint_->GetPosition().z());
  printf("(%10f,%10f,%10f)",
         postStepPoint_->GetPosition().x(),
         postStepPoint_->GetPosition().y(),
         postStepPoint_->GetPosition().z());
  LogDebug("TotemRP") << " positions "
                      << "(" << postStepPoint_->GetPosition().x() << "," << postStepPoint_->GetPosition().y() << ","
                      << postStepPoint_->GetPosition().z() << ")"
                      << " For Track  " << theTrack_->GetTrackID() << " which is a "
                      << theTrack_->GetDefinition()->GetParticleName();

  if (theTrack_->GetTrackID() == 1) {
    LogDebug("TotemRP") << " primary particle ";
  } else {
    LogDebug("TotemRP") << " daughter of part. " << theTrack_->GetParentID();
  }

  LogDebug("TotemRP") << " and created by ";

  if (theTrack_->GetCreatorProcess() != nullptr)
    LogDebug("TotemRP") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("TotemRP") << "NO process";

  LogDebug("TotemRP") << std::endl;
}

bool TotemRPSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (aStep == nullptr) {
    return true;
  } else {
    stepInfo(aStep);

    createNewHit();
    return true;
  }
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

  G4String name_ = currentPV_->GetName();
  name_.assign(name_, 0, 4);
  G4String particleType = theTrack_->GetDefinition()->GetParticleName();
  tSlice_ = (postStepPoint_->GetGlobalTime()) / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);

  if (verbosity_)
    LogDebug("TotemRP") << "UNIT " << unitID_ << std::endl;

  primaryID_ = theTrack_->GetTrackID();

  Pabs_ = (aStep->GetPreStepPoint()->GetMomentum().mag()) / GeV;
  thePx_ = (aStep->GetPreStepPoint()->GetMomentum().x()) / GeV;
  thePy_ = (aStep->GetPreStepPoint()->GetMomentum().y()) / GeV;
  thePz_ = (aStep->GetPreStepPoint()->GetMomentum().z()) / GeV;

  Tof_ = aStep->GetPostStepPoint()->GetGlobalTime() / nanosecond;
  Eloss_ = aStep->GetTotalEnergyDeposit() / GeV;
  ParticleType_ = theTrack_->GetDefinition()->GetPDGEncoding();

  //corrected phi and theta treatment
  G4ThreeVector gmd = aStep->GetPreStepPoint()->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd = ((G4TouchableHistory*)(aStep->GetPreStepPoint()->GetTouchable()))
                          ->GetHistory()
                          ->GetTopTransform()
                          .TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  ThetaAtEntry_ = lnmd.theta();
  PhiAtEntry_ = lnmd.phi();

  if (isPrimary(theTrack_))
    ParentId_ = 0;
  else
    ParentId_ = theTrack_->GetParentID();

  Vx_ = theTrack_->GetVertexPosition().x() / mm;
  Vy_ = theTrack_->GetVertexPosition().y() / mm;
  Vz_ = theTrack_->GetVertexPosition().z() / mm;
}

uint32_t TotemRPSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void TotemRPSD::storeHit(TotemRPG4Hit* hit) {
  if (hit == nullptr) {
    if (verbosity_)
      LogDebug("TotemRP") << "TotemRPSD: hit to be stored is NULL !!" << std::endl;
    return;
  }
  theHC_->insert(hit);
}

void TotemRPSD::createNewHit() {
  // Protect against creating hits in detectors not inserted
  double outrangeX = hitPoint_.x();
  double outrangeY = hitPoint_.y();
  if (fabs(outrangeX) > rp_garage_position_)
    return;
  if (fabs(outrangeY) > rp_garage_position_)
    return;
  // end protection

  currentHit_ = new TotemRPG4Hit;
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

  currentHit_->setEntry(hitPoint_);
  currentHit_->setExit(exitPoint_);
  currentHit_->setLocalEntry(theLocalEntryPoint_);
  currentHit_->setLocalExit(theLocalExitPoint_);

  currentHit_->setParentId(ParentId_);
  currentHit_->setVx(Vx_);
  currentHit_->setVy(Vy_);
  currentHit_->setVz(Vz_);

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
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries() && j < maxTotemHits_; j++) {
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
  summarize();
}

void TotemRPSD::summarize() {}

void TotemRPSD::clear() {}

void TotemRPSD::DrawAll() {}

void TotemRPSD::PrintAll() {
  LogDebug("TotemRP") << "TotemRPSD: Collection " << theHC_->GetName() << std::endl;
  theHC_->PrintAllHits();
}

void TotemRPSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave_->name() == n) {
    c = slave_->hits();
  }
}
void TotemRPSD::setNumberingScheme(TotemRPVDetectorOrganization* scheme) {
  if (scheme) {
    LogDebug("TotemRP") << "TotemRPSD: updates numbering scheme for " << GetName();
    numberingScheme_.reset(scheme);
  }
}
void TotemRPSD::update(const BeginOfEvent* i) {
  clearHits();
  eventno_ = (*i)()->GetEventID();
}

void TotemRPSD::update(const ::EndOfEvent*) {}

void TotemRPSD::clearHits() { slave_->Initialize(); }

bool TotemRPSD::isPrimary(const G4Track* track) {
  TrackInformation* info = dynamic_cast<TrackInformation*>(track->GetUserInformation());
  return info && info->isPrimary();
}
