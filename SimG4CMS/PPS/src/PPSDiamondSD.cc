///////////////////////////////////////////////////////////////////////////////
//Author: Seyed Mohsen Etesami
// setesami@cern.ch
// 2016 Nov
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/PPS/interface/PPSDiamondSD.h"
#include "SimG4CMS/PPS/interface/PPSDiamondOrganization.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <iostream>
#include <vector>
#include <string>

PPSDiamondSD::PPSDiamondSD(const std::string& pname,
                           const SensitiveDetectorCatalog& clg,
                           edm::ParameterSet const& p,
                           const SimTrackManager* manager)
    : SensitiveTkDetector(pname, clg) {
  collectionName.insert(pname);
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("PPSDiamondSD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");

  LogDebug("PPSSimDiamond") << "*******************************************************\n"
                            << "*                                                     *\n"
                            << "* Constructing a PPSDiamondSD  with name " << pname << "\n"
                            << "*                                                     *\n"
                            << "*******************************************************"
                            << "\n";

  slave_ = std::make_unique<TrackingSlaveSD>(pname);

  if (pname == "CTPPSTimingHits") {
    numberingScheme_ = std::make_unique<PPSDiamondOrganization>();
    edm::LogVerbatim("PPSSimDiamond") << "Find CTPPSTimingHits as name";
  } else {
    edm::LogError("PPSSimDiamond") << "PPSDiamondSD: ReadoutName " << pname << " not supported";
  }

  edm::LogVerbatim("PPSSimDiamond") << "PPSDiamondSD: Instantiation completed for " << pname;
}

PPSDiamondSD::~PPSDiamondSD() {}

void PPSDiamondSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("PPSSimDiamond") << "PPSDiamondSD : Initialize called for " << GetName();

  theHC_ = new PPSDiamondG4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(GetName(), collectionName[0]);
  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);
}

void PPSDiamondSD::printHitInfo() {
  LogDebug("PPSSimDiamond") << theTrack_->GetDefinition()->GetParticleName() << " PPS_Timing_SD CreateNewHit for"
                            << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo() << " Unit "
                            << unitID_ << "\n";
  LogDebug("PPSSimDiamond") << " primary " << primaryID_ << " time slice " << tSliceID_ << " of energy "
                            << theTrack_->GetTotalEnergy() << " Eloss=" << eloss_ << " position pre: " << hitPoint_
                            << " post: " << exitPoint_;
  LogDebug("PPSSimDiamond") << " positions "
                            << "(" << postStepPoint_->GetPosition().x() << "," << postStepPoint_->GetPosition().y()
                            << "," << postStepPoint_->GetPosition().z() << ")"
                            << " For Track  " << theTrack_->GetTrackID() << " which is a "
                            << theTrack_->GetDefinition()->GetParticleName() << " ParentID is "
                            << theTrack_->GetParentID() << "\n";

  if (parentID_ == 0) {
    LogDebug("PPSSimDiamond") << " primary particle ";
  } else {
    LogDebug("PPSSimDiamond") << " daughter of part. " << parentID_;
  }

  LogDebug("PPSSimDiamond") << " and created by ";

  if (theTrack_->GetCreatorProcess() != nullptr)
    LogDebug("PPSSimDiamond") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("PPSSimDiamond") << "NO process";
}

bool PPSDiamondSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  eloss_ = aStep->GetTotalEnergyDeposit();
  if (eloss_ > 0.0) {
    eloss_ /= GeV;
    stepInfo(aStep);
    if (theTrack_->GetDefinition()->GetPDGEncoding() == 2212) {
      LogDebug("PPSSimDiamond") << "PPSSimDiamond : eloss=" << eloss_;
      importInfoToHit();  //in addition to import info to hit it STORE hit as well
      LogDebug("PPSSimDiamond") << " information imported to the hit ";
    }
  }
  return true;
}

void PPSDiamondSD::stepInfo(const G4Step* aStep) {
  theTrack_ = aStep->GetTrack();
  preStepPoint_ = aStep->GetPreStepPoint();
  postStepPoint_ = aStep->GetPostStepPoint();
  hitPoint_ = preStepPoint_->GetPosition();
  exitPoint_ = postStepPoint_->GetPosition();
  currentPV_ = preStepPoint_->GetPhysicalVolume();
  theLocalEntryPoint_ = setToLocal(hitPoint_);
  theLocalExitPoint_ = setToLocal(exitPoint_);
  tof_ = preStepPoint_->GetGlobalTime() / nanosecond;
  incidentEnergy_ = preStepPoint_->GetTotalEnergy() / eV;
  tSlice_ = postStepPoint_->GetGlobalTime() / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);

  if (verbosity_)
    LogDebug("PPSSimDiamond") << "UNIT " << unitID_;

  primaryID_ = theTrack_->GetTrackID();
  parentID_ = theTrack_->GetParentID();

  incidentEnergy_ = theTrack_->GetTotalEnergy() / GeV;

  pabs_ = preStepPoint_->GetMomentum().mag() / GeV;
  thePx_ = preStepPoint_->GetMomentum().x() / GeV;
  thePy_ = preStepPoint_->GetMomentum().y() / GeV;
  thePz_ = preStepPoint_->GetMomentum().z() / GeV;
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

uint32_t PPSDiamondSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void PPSDiamondSD::storeHit(PPSDiamondG4Hit* hit) {
  if (hit == nullptr) {
    if (verbosity_)
      LogDebug("PPSSimDiamond") << "PPSDiamond: hit to be stored is NULL !!";
    return;
  }

  theHC_->insert(hit);
}

void PPSDiamondSD::importInfoToHit() {
  currentHit_ = new PPSDiamondG4Hit;
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
  currentHit_->setGlobalTimehit(tof_);

  storeHit(currentHit_);
  LogDebug("PPSSimDiamond") << "STORED HIT IN: " << unitID_ << "\n";
}

G4ThreeVector PPSDiamondSD::setToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint_->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);

  return localPoint;
}

void PPSDiamondSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries(); ++j) {
    PPSDiamondG4Hit* aHit = (*theHC_)[j];

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

void PPSDiamondSD::PrintAll() {
  LogDebug("PPSSimDiamond") << "PPSDiamond: Collection " << theHC_->GetName() << "\n";
  theHC_->PrintAllHits();
}

void PPSDiamondSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave_->name() == n)
    c = slave_->hits();
}

void PPSDiamondSD::update(const BeginOfEvent* i) {
  LogDebug("PPSSimDiamond") << " Dispatched BeginOfEvent !"
                            << "\n";
  clearHits();
  eventno_ = (*i)()->GetEventID();
}

void PPSDiamondSD::update(const ::EndOfEvent*) {}

void PPSDiamondSD::clearHits() { slave_->Initialize(); }
