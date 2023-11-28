#include "SimG4CMS/Muon/interface/MuonSensitiveDetector.h"
#include "SimG4CMS/Muon/interface/MuonSlaveSD.h"
#include "SimG4CMS/Muon/interface/MuonEndcapFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonRPCFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonGEMFrameRotation.h"
#include "SimG4CMS/Muon/interface/MuonME0FrameRotation.h"
#include "Geometry/MuonNumbering/interface/MuonSubDetector.h"

#include "SimG4CMS/Muon/interface/SimHitPrinter.h"
#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonSimHitNumberingScheme.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "G4VProcess.hh"
#include "G4EventManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//#define EDM_ML_DEBUG

MuonSensitiveDetector::MuonSensitiveDetector(const std::string& name,
                                             const MuonOffsetMap* offmap,
                                             const MuonGeometryConstants& constants,
                                             const SensitiveDetectorCatalog& clg,
                                             edm::ParameterSet const& p,
                                             const SimTrackManager* manager)
    : SensitiveTkDetector(name, clg),
      thePV(nullptr),
      theHit(nullptr),
      theDetUnitId(0),
      newDetUnitId(0),
      theTrackID(0),
      thePrinter(nullptr),
      theManager(manager) {
  // Here simply create 1 MuonSlaveSD for the moment
  //
  bool dd4hep = p.getParameter<bool>("g4GeometryDD4hepSource");
  edm::ParameterSet muonSD = p.getParameter<edm::ParameterSet>("MuonSD");
  printHits_ = muonSD.getParameter<bool>("PrintHits");
  ePersistentCutGeV_ = muonSD.getParameter<double>("EnergyThresholdForPersistency") / CLHEP::GeV;  //Default 1. GeV
  allMuonsPersistent_ = muonSD.getParameter<bool>("AllMuonsPersistent");
  haveDemo_ = muonSD.getParameter<bool>("HaveDemoChambers");
  demoGEM_ = muonSD.getParameter<bool>("UseDemoHitGEM");
  demoRPC_ = muonSD.getParameter<bool>("UseDemoHitRPC");

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << "create MuonSubDetector " << name << " with dd4hep flag " << dd4hep
                              << " Flags for Demonstration chambers " << haveDemo_ << " for GEM " << demoGEM_
                              << " for RPC " << demoRPC_;
#endif
  detector = new MuonSubDetector(name);

  G4String sdet = "unknown";
  if (detector->isEndcap()) {
    theRotation = new MuonEndcapFrameRotation();
    sdet = "Endcap";
  } else if (detector->isRPC()) {
    theRotation = new MuonRPCFrameRotation(constants, offmap, dd4hep);
    sdet = "RPC";
  } else if (detector->isGEM()) {
    theRotation = new MuonGEMFrameRotation(constants);
    sdet = "GEM";
  } else if (detector->isME0()) {
    theRotation = new MuonME0FrameRotation(constants);
    sdet = "ME0";
  } else {
    theRotation = new MuonFrameRotation();
    if (detector->isBarrel())
      sdet = "Barrel";
  }
  slaveMuon = new MuonSlaveSD(detector, theManager);
  numbering = new MuonSimHitNumberingScheme(detector, constants);
  g4numbering = new MuonG4Numbering(constants, offmap, dd4hep);

  if (printHits_) {
    thePrinter = new SimHitPrinter("HitPositionOSCAR.dat");
  }

  edm::LogVerbatim("MuonSim") << " of type " << sdet << " <" << GetName() << "> EnergyThresholdForPersistency(GeV) "
                              << ePersistentCutGeV_ / CLHEP::GeV << " allMuonsPersistent: " << allMuonsPersistent_;

  theG4ProcessTypeEnumerator = new G4ProcessTypeEnumerator;
}

MuonSensitiveDetector::~MuonSensitiveDetector() {
  delete g4numbering;
  delete numbering;
  delete slaveMuon;
  delete theRotation;
  delete detector;
  delete theG4ProcessTypeEnumerator;
}

void MuonSensitiveDetector::update(const BeginOfEvent* i) {
  clearHits();
  //----- Initialize variables to check if two steps belong to same hit
  thePV = nullptr;
  theDetUnitId = 0;
  theTrackID = 0;
}

void MuonSensitiveDetector::clearHits() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << "MuonSensitiveDetector::clearHits";
#endif
  slaveMuon->Initialize();
}

bool MuonSensitiveDetector::ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << " MuonSensitiveDetector::ProcessHits " << InitialStepPosition(aStep, WorldCoordinates);
#endif

  if (aStep->GetTotalEnergyDeposit() > 0.) {
    newDetUnitId = setDetUnitId(aStep);
#ifdef EDM_ML_DEBUG
    G4VPhysicalVolume* vol = aStep->GetPreStepPoint()->GetTouchable()->GetVolume(0);
    std::string namx = static_cast<std::string>(vol->GetName());
    std::string name = namx.substr(0, 2);
    if (name == "RE")
      edm::LogVerbatim("MuonSim") << "DETID " << namx << " " << RPCDetId(newDetUnitId);
#endif
    if (newHit(aStep)) {
      saveHit();
      createHit(aStep);
    } else {
      updateHit(aStep);
    }
    thePV = aStep->GetPreStepPoint()->GetPhysicalVolume();
    theTrackID = aStep->GetTrack()->GetTrackID();
    theDetUnitId = newDetUnitId;
  }
  return true;
}

uint32_t MuonSensitiveDetector::setDetUnitId(const G4Step* aStep) {
  MuonBaseNumber num = g4numbering->PhysicalVolumeToBaseNumber(aStep);

#ifdef EDM_ML_DEBUG
  std::stringstream MuonBaseNumber;
  MuonBaseNumber << "MuonNumbering :: number of levels = " << num.getLevels() << std::endl;
  MuonBaseNumber << "Level \t SuperNo \t BaseNo" << std::endl;
  for (int level = 1; level <= num.getLevels(); level++) {
    MuonBaseNumber << level << " \t " << num.getSuperNo(level) << " \t " << num.getBaseNo(level) << std::endl;
  }
  std::string MuonBaseNumbr = MuonBaseNumber.str();

  edm::LogVerbatim("MuonSim") << "MuonSensitiveDetector::setDetUnitId :: " << MuonBaseNumbr;
  edm::LogVerbatim("MuonSim") << "MuonSensitiveDetector::setDetUnitId :: MuonDetUnitId = "
                              << (numbering->baseNumberToUnitNumber(num));
#endif
  return numbering->baseNumberToUnitNumber(num);
}

bool MuonSensitiveDetector::newHit(const G4Step* aStep) {
  return (!theHit || (aStep->GetTrack()->GetTrackID() != theTrackID) ||
          (aStep->GetPreStepPoint()->GetPhysicalVolume() != thePV) || newDetUnitId != theDetUnitId);
}

void MuonSensitiveDetector::createHit(const G4Step* aStep) {
  Local3DPoint theEntryPoint;
  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    // 1 levels up
    theEntryPoint = cmsUnits(theRotation->transformPoint(InitialStepPositionVsParent(aStep, 1), aStep));
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPositionVsParent(aStep, 1), aStep));
  } else if (detector->isEndcap()) {
    // save local z at current level
    theEntryPoint = theRotation->transformPoint(InitialStepPosition(aStep, LocalCoordinates), aStep);
    theExitPoint = theRotation->transformPoint(FinalStepPosition(aStep, LocalCoordinates), aStep);
    float zentry = theEntryPoint.z();
    float zexit = theExitPoint.z();
    // 4 levels up
    Local3DPoint tempEntry = theRotation->transformPoint(InitialStepPositionVsParent(aStep, 4), aStep);
    Local3DPoint tempExit = theRotation->transformPoint(FinalStepPositionVsParent(aStep, 4), aStep);
    // reset local z from z wrt deep-parent volume to z wrt low-level volume
    theEntryPoint = cmsUnits(Local3DPoint(tempEntry.x(), tempEntry.y(), zentry));
    theExitPoint = cmsUnits(Local3DPoint(tempExit.x(), tempExit.y(), zexit));
  } else {
    theEntryPoint = cmsUnits(theRotation->transformPoint(InitialStepPosition(aStep, LocalCoordinates), aStep));
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPosition(aStep, LocalCoordinates), aStep));
  }

  const G4Track* theTrack = aStep->GetTrack();
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  float thePabs = preStepPoint->GetMomentum().mag() / CLHEP::GeV;
  float theTof = preStepPoint->GetGlobalTime() / CLHEP::nanosecond;
  float theEnergyLoss = aStep->GetTotalEnergyDeposit() / CLHEP::GeV;
  int theParticleType = G4TrackToParticleID::particleID(theTrack);

  theDetUnitId = newDetUnitId;
  thePV = preStepPoint->GetPhysicalVolume();
  theTrackID = theTrack->GetTrackID();

  // convert momentum direction it to local frame
  const G4ThreeVector& gmd = preStepPoint->GetMomentumDirection();
  G4ThreeVector lmd = static_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable())
                          ->GetHistory()
                          ->GetTopTransform()
                          .TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  lnmd = theRotation->transformPoint(lnmd, aStep);
  float theThetaAtEntry = lnmd.theta();
  float thePhiAtEntry = lnmd.phi();

  theHit = new UpdatablePSimHit(theEntryPoint,
                                theExitPoint,
                                thePabs,
                                theTof,
                                theEnergyLoss,
                                theParticleType,
                                theDetUnitId,
                                theTrackID,
                                theThetaAtEntry,
                                thePhiAtEntry,
                                theG4ProcessTypeEnumerator->processId(theTrack->GetCreatorProcess()));

  // Make track persistent
  int thePID = std::abs(theTrack->GetDefinition()->GetPDGEncoding());
  //---VI - in parameters cut in energy is declared but applied to momentum
  if (thePabs > ePersistentCutGeV_ || (thePID == 13 && allMuonsPersistent_)) {
    TrackInformation* info = cmsTrackInformation(theTrack);
    info->setStoreTrack();
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << "=== NEW Muon hit for " << GetName() << " Edep(GeV)= " << theEnergyLoss << " "
                              << thePV->GetLogicalVolume()->GetName();
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  G4String sss = "";
  if (p)
    sss += " POST PROCESS: " + p->GetProcessName();
  if (p2)
    sss += ";  PRE  PROCESS: " + p2->GetProcessName();
  if (!sss.empty())
    edm::LogVerbatim("MuonSim") << sss;
  edm::LogVerbatim("MuonSim") << " theta= " << theThetaAtEntry << " phi= " << thePhiAtEntry << " Pabs(GeV/c)  "
                              << thePabs << " Eloss(GeV)= " << theEnergyLoss << " Tof(ns)=  " << theTof
                              << " trackID= " << theTrackID << " detID= " << theDetUnitId << "\n Local:  entry "
                              << theEntryPoint << " exit " << theExitPoint << " delta "
                              << (theExitPoint - theEntryPoint) << "\n Global: entry "
                              << aStep->GetPreStepPoint()->GetPosition() << " exit "
                              << aStep->GetPostStepPoint()->GetPosition();
#endif
}

void MuonSensitiveDetector::updateHit(const G4Step* aStep) {
  Local3DPoint theExitPoint;

  if (detector->isBarrel()) {
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPositionVsParent(aStep, 1), aStep));
  } else if (detector->isEndcap()) {
    // save local z at current level
    theExitPoint = theRotation->transformPoint(FinalStepPosition(aStep, LocalCoordinates), aStep);
    float zexit = theExitPoint.z();
    Local3DPoint tempExitPoint = theRotation->transformPoint(FinalStepPositionVsParent(aStep, 4), aStep);
    theExitPoint = cmsUnits(Local3DPoint(tempExitPoint.x(), tempExitPoint.y(), zexit));
  } else {
    theExitPoint = cmsUnits(theRotation->transformPoint(FinalStepPosition(aStep, LocalCoordinates), aStep));
  }

  float theEnergyLoss = aStep->GetTotalEnergyDeposit() / CLHEP::GeV;

  theHit->updateExitPoint(theExitPoint);
  theHit->addEnergyLoss(theEnergyLoss);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << "=== NEW Update muon hit for " << GetName() << " Edep(GeV)= " << theEnergyLoss << " "
                              << thePV->GetLogicalVolume()->GetName();
  const G4VProcess* p = aStep->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VProcess* p2 = aStep->GetPreStepPoint()->GetProcessDefinedStep();
  G4String sss = "";
  if (p)
    sss += " POST PROCESS: " + p->GetProcessName();
  if (p2)
    sss += ";  PRE  PROCESS: " + p2->GetProcessName();
  if (!sss.empty())
    edm::LogVerbatim("MuonSim") << sss;
  edm::LogVerbatim("MuonSim") << " delEloss(GeV)= " << theEnergyLoss
                              << " Tof(ns)=  " << aStep->GetPreStepPoint()->GetGlobalTime() / CLHEP::nanosecond
                              << " trackID= " << theTrackID << " detID= " << theDetUnitId << " exit " << theExitPoint;
#endif
}

void MuonSensitiveDetector::saveHit() {
  if (theHit) {
    if (acceptHit(theHit->detUnitId())) {
      if (printHits_) {
        thePrinter->startNewSimHit(detector->name());
        thePrinter->printId(theHit->detUnitId());
        thePrinter->printLocal(theHit->entryPoint(), theHit->exitPoint());
      }
      // hit is included into hit collection
      slaveMuon->processHits(*theHit);
    }
    delete theHit;
    theHit = nullptr;
  }
}

void MuonSensitiveDetector::EndOfEvent(G4HCofThisEvent*) { saveHit(); }

void MuonSensitiveDetector::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slaveMuon->name() == hname) {
    cc = slaveMuon->hits();
  }
}

Local3DPoint MuonSensitiveDetector::InitialStepPositionVsParent(const G4Step* currentStep, G4int levelsUp) {
  const G4StepPoint* preStepPoint = currentStep->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = preStepPoint->GetPosition();

  const G4TouchableHistory* theTouchable = (const G4TouchableHistory*)(preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates =
      theTouchable->GetHistory()->GetTransform(depth - levelsUp).TransformPoint(globalCoordinates);

  return ConvertToLocal3DPoint(localCoordinates);
}

Local3DPoint MuonSensitiveDetector::FinalStepPositionVsParent(const G4Step* currentStep, G4int levelsUp) {
  const G4StepPoint* postStepPoint = currentStep->GetPostStepPoint();
  const G4StepPoint* preStepPoint = currentStep->GetPreStepPoint();
  const G4ThreeVector& globalCoordinates = postStepPoint->GetPosition();

  const G4TouchableHistory* theTouchable = (const G4TouchableHistory*)(preStepPoint->GetTouchable());

  G4int depth = theTouchable->GetHistory()->GetDepth();
  G4ThreeVector localCoordinates =
      theTouchable->GetHistory()->GetTransform(depth - levelsUp).TransformPoint(globalCoordinates);

  return ConvertToLocal3DPoint(localCoordinates);
}

bool MuonSensitiveDetector::acceptHit(uint32_t id) {
  if (id == 0) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonSim") << "DetId " << id << " Flag " << false;
#endif
    return false;
  }
  bool flag(true);
  if (haveDemo_) {
    int subdet = DetId(id).subdetId();
    if (subdet == MuonSubdetId::GEM) {
      if (GEMDetId(id).station() == 2)
        flag = demoGEM_;
    } else if (subdet == MuonSubdetId::RPC) {
      if ((RPCDetId(id).region() != 0) && (RPCDetId(id).ring() == 1) && (RPCDetId(id).station() > 2))
        flag = demoRPC_;
    }
  }
#ifdef EDM_ML_DEBUG
  int subdet = DetId(id).subdetId();
  if (subdet == MuonSubdetId::RPC)
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " RPC " << RPCDetId(id) << " Flag "
                                << flag;
  else if (subdet == MuonSubdetId::GEM)
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " GEM " << GEMDetId(id) << " Flag "
                                << flag;
  else if (subdet == MuonSubdetId::ME0)
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " " << ME0DetId(id) << " Flag " << flag;
  else if (subdet == MuonSubdetId::CSC)
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " CSC Flag " << flag;
  else if (subdet == MuonSubdetId::DT)
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " DT Flag " << flag;
  else
    edm::LogVerbatim("MuonSim") << "DetId " << std::hex << id << std::dec << " Unknown Flag " << flag;
#endif
  return flag;
}
