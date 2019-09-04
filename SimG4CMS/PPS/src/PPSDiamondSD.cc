///////////////////////////////////////////////////////////////////////////////
//Author: Seyed Mohsen Etesami
// setesami@cern.ch
// 2016 Nov
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/PPS/interface/PPSDiamondSD.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

PPSDiamondSD::PPSDiamondSD(const std::string& name_,
                           const edm::EventSetup& es,
                           const SensitiveDetectorCatalog& clg,
                           edm::ParameterSet const& p,
                           const SimTrackManager* manager)
    : SensitiveTkDetector(name_, es, clg, p),
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
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("PPSDiamondSD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");

  LogDebug("PPSSimDiamond") << "*******************************************************\n"
                            << "*                                                     *\n"
                            << "* Constructing a PPSDiamondSD  with name " << name_ << "\n"
                            << "*                                                     *\n"
                            << "*******************************************************"
                            << "\n";

  slave_ = std::make_unique<TrackingSlaveSD>(name_);

  if (name_ == "CTPPSTimingHits") {
    numberingScheme_ = std::make_unique<PPSDiamondNumberingScheme>();
    edm::LogInfo("PPSSimDiamond") << "Find CTPPSDiamondHits as name";
  } else {
    edm::LogWarning("PPSSimDiamond") << "PPSDiamondSD: ReadoutName not supported\n";
  }

  edm::LogInfo("PPSSimDiamond") << "PPSDiamondSD: Instantiation completed";
}

PPSDiamondSD::~PPSDiamondSD() {}

void PPSDiamondSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("PPSSimDiamond") << "PPSDiamondSD : Initialize called for " << name_;

  theHC_ = new PPSDiamondG4HitCollection(name_, collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(name_, collectionName[0]);
  if (hcID_ < 0)
    hcID_ = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID_, theHC_);
}

void PPSDiamondSD::printHitInfo() {
  LogDebug("PPSSimDiamond") << theTrack_->GetDefinition()->GetParticleName() << " PPS_Timing_SD CreateNewHit for"
                            << " PV " << currentPV_->GetName() << " PVid = " << currentPV_->GetCopyNo() << " Unit "
                            << unitID_ << "\n";
  LogDebug("PPSSimDiamond") << " primary " << primaryID_ << " time slice " << tSliceID_ << " of energy "
                            << theTrack_->GetTotalEnergy() << " Eloss_ " << Eloss_ << " positions "
                            << "\n";
  printf(" PreStepPoint(%10f,%10f,%10f)",
         preStepPoint_->GetPosition().x(),
         preStepPoint_->GetPosition().y(),
         preStepPoint_->GetPosition().z());
  printf(" PosStepPoint(%10f,%10f,%10f)\n",
         postStepPoint_->GetPosition().x(),
         postStepPoint_->GetPosition().y(),
         postStepPoint_->GetPosition().z());
  LogDebug("PPSSimDiamond") << " positions "
                            << "(" << postStepPoint_->GetPosition().x() << "," << postStepPoint_->GetPosition().y()
                            << "," << postStepPoint_->GetPosition().z() << ")"
                            << " For Track  " << theTrack_->GetTrackID() << " which is a "
                            << theTrack_->GetDefinition()->GetParticleName() << " ParentID is "
                            << theTrack_->GetParentID() << "\n";

  if (theTrack_->GetTrackID() == 1) {
    LogDebug("PPSSimDiamond") << " primary particle ";
  } else {
    LogDebug("PPSSimDiamond") << " daughter of part. " << theTrack_->GetParentID();
  }

  LogDebug("PPSSimDiamond") << " and created by ";

  if (theTrack_->GetCreatorProcess() != nullptr)
    LogDebug("PPSSimDiamond") << theTrack_->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("PPSSimDiamond") << "NO process";

  LogDebug("PPSSimDiamond") << "\n";
}

bool PPSDiamondSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (aStep == nullptr) {
    LogDebug("PPSSimDiamond") << " There is no hit to process ";
    return true;
  } else {
    LogDebug("PPSSimDiamond") << "*******************************************************\n"
                              << "*                                                     *\n"
                              << "* PPS Diamond Hit initialized  with name " << name_ << "\n"
                              << "*                                                     *\n"
                              << "*******************************************************"
                              << "\n";

    stepInfo(aStep);

    if (theTrack_->GetDefinition()->GetPDGEncoding() == 2212) {
      importInfoToHit();  //in addtion to import info to hit it STORE hit as well
      LogDebug("PPSSimDiamond") << " information imported to the hit ";
    }

    return true;
  }
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
  theglobaltimehit_ = preStepPoint_->GetGlobalTime() / nanosecond;
  incidentEnergy_ = (aStep->GetPreStepPoint()->GetTotalEnergy() / eV);
  tSlice_ = (postStepPoint_->GetGlobalTime()) / nanosecond;
  tSliceID_ = (int)tSlice_;
  unitID_ = setDetUnitId(aStep);

  if (verbosity_)
    LogDebug("PPSSimDiamond") << "UNIT " << unitID_ << "\n";

  primaryID_ = theTrack_->GetTrackID();
  Pabs_ = (aStep->GetPreStepPoint()->GetMomentum().mag()) / GeV;
  thePx_ = (aStep->GetPreStepPoint()->GetMomentum().x()) / GeV;
  thePy_ = (aStep->GetPreStepPoint()->GetMomentum().y()) / GeV;
  thePz_ = (aStep->GetPreStepPoint()->GetMomentum().z()) / GeV;
  Tof_ = aStep->GetPreStepPoint()->GetGlobalTime() / nanosecond;
  Eloss_ = (aStep->GetPreStepPoint()->GetTotalEnergy() / eV);  //pps added
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

uint32_t PPSDiamondSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->unitID(aStep));
}

void PPSDiamondSD::storeHit(PPSDiamondG4Hit* hit) {
  if (hit == nullptr) {
    if (verbosity_)
      LogDebug("PPSSimDiamond") << "PPSDiamond: hit to be stored is NULL !!"
                                << "\n";
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
  currentHit_->setGlobalTimehit(Globaltimehit_);

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
  for (unsigned int j = 0; j < (unsigned int)theHC_->entries() && j < maxDiamondHits_; j++) {
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
  summarize();
}

void PPSDiamondSD::summarize() {}

void PPSDiamondSD::clear() {}

void PPSDiamondSD::DrawAll() {}

void PPSDiamondSD::PrintAll() {
  LogDebug("PPSSimDiamond") << "PPSDiamond: Collection " << theHC_->GetName() << "\n";
  theHC_->PrintAllHits();
}

void PPSDiamondSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave_->name() == n)
    c = slave_->hits();
}
void PPSDiamondSD::setNumberingScheme(PPSVDetectorOrganization* scheme) {
  if (scheme) {
    LogDebug("PPSDiamond") << "PPSDiamondSD: updates numbering scheme for " << GetName();
    numberingScheme_.reset(scheme);
  }
}
void PPSDiamondSD::update(const BeginOfEvent* i) {
  LogDebug("PPSSimDiamond") << " Dispatched BeginOfEvent !"
                            << "\n";
  clearHits();
  eventno_ = (*i)()->GetEventID();
}

void PPSDiamondSD::update(const ::EndOfEvent*) {}

void PPSDiamondSD::clearTrack(G4Track* track) { track->SetTrackStatus(fStopAndKill); }

void PPSDiamondSD::clearHits() { slave_->Initialize(); }

bool PPSDiamondSD::isPrimary(const G4Track* track) {
  TrackInformation* info = dynamic_cast<TrackInformation*>(track->GetUserInformation());
  return info && info->isPrimary();
}
