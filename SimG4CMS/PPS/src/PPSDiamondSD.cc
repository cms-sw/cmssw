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

PPSDiamondSD::PPSDiamondSD(const std::string& name,
                           const edm::EventSetup& es,
                           const SensitiveDetectorCatalog& clg,
                           edm::ParameterSet const& p,
                           const SimTrackManager* manager)
    : SensitiveTkDetector(name, es, clg, p),
      numberingScheme(nullptr),
      hcID(-1),
      theHC(nullptr),
      currentHit(nullptr),
      theTrack(nullptr),
      currentPV(nullptr),
      unitID(0),
      preStepPoint(nullptr),
      postStepPoint(nullptr),
      eventno(0) {
  collectionName.insert(name);
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("PPSDiamondSD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");

  LogDebug("PPSSimDiamond") << "*******************************************************\n"
                            << "*                                                     *\n"
                            << "* Constructing a PPSDiamondSD  with name " << name << "\n"
                            << "*                                                     *\n"
                            << "*******************************************************"
                            << "\n";

  slave = new TrackingSlaveSD(name);

  if (name == "CTPPSTimingHits") {
    numberingScheme = dynamic_cast<PPSVDetectorOrganization*>(new PPSDiamondNumberingScheme());
    edm::LogInfo("PPSSimDiamond") << "Find CTPPSDiamondHits as name";
  } else {
    edm::LogWarning("PPSSimDiamond") << "PPSDiamondSD: ReadoutName not supported\n";
  }

  edm::LogInfo("PPSSimDiamond") << "PPSDiamondSD: Instantiation completed";
}

PPSDiamondSD::~PPSDiamondSD() {
  delete slave;
  delete numberingScheme;
}

void PPSDiamondSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("PPSSimDiamond") << "PPSDiamondSD : Initialize called for " << name;

  theHC = new PPSDiamondG4HitCollection(name, collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(name, collectionName[0]);
  if (hcID < 0)
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
}

void PPSDiamondSD::Print_Hit_Info() {
  LogDebug("PPSSimDiamond") << theTrack->GetDefinition()->GetParticleName() << " PPS_Timing_SD CreateNewHit for"
                            << " PV " << currentPV->GetName() << " PVid = " << currentPV->GetCopyNo() << " Unit "
                            << unitID << "\n";
  LogDebug("PPSSimDiamond") << " primary " << primaryID << " time slice " << tSliceID << " of energy "
                            << theTrack->GetTotalEnergy() << " Eloss " << Eloss << " positions "
                            << "\n";
  printf(" PreStepPoint(%10f,%10f,%10f)",
         preStepPoint->GetPosition().x(),
         preStepPoint->GetPosition().y(),
         preStepPoint->GetPosition().z());
  printf(" PosStepPoint(%10f,%10f,%10f)\n",
         postStepPoint->GetPosition().x(),
         postStepPoint->GetPosition().y(),
         postStepPoint->GetPosition().z());
  LogDebug("PPSSimDiamond") << " positions "
                            << "(" << postStepPoint->GetPosition().x() << "," << postStepPoint->GetPosition().y() << ","
                            << postStepPoint->GetPosition().z() << ")"
                            << " For Track  " << theTrack->GetTrackID() << " which is a "
                            << theTrack->GetDefinition()->GetParticleName() << " ParentID is "
                            << theTrack->GetParentID() << "\n";

  if (theTrack->GetTrackID() == 1) {
    LogDebug("PPSSimDiamond") << " primary particle ";
  } else {
    LogDebug("PPSSimDiamond") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("PPSSimDiamond") << " and created by ";

  if (theTrack->GetCreatorProcess() != nullptr)
    LogDebug("PPSSimDiamond") << theTrack->GetCreatorProcess()->GetProcessName();
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
                              << "* PPS Diamond Hit initialized  with name " << name << "\n"
                              << "*                                                     *\n"
                              << "*******************************************************"
                              << "\n";

    GetStepInfo(aStep);

    if (theTrack->GetDefinition()->GetPDGEncoding() == 2212) {
      //Print_Hit_Info();
      ImportInfotoHit();  //in addtion to import info to hit it STORE hit as well
      LogDebug("PPSSimDiamond") << " information imported to the hit ";
    }

    return true;
  }
}

void PPSDiamondSD::GetStepInfo(const G4Step* aStep) {
  theTrack = aStep->GetTrack();
  preStepPoint = aStep->GetPreStepPoint();
  postStepPoint = aStep->GetPostStepPoint();
  hitPoint = preStepPoint->GetPosition();
  exitPoint = postStepPoint->GetPosition();
  currentPV = preStepPoint->GetPhysicalVolume();
  theLocalEntryPoint = SetToLocal(hitPoint);
  theLocalExitPoint = SetToLocal(exitPoint);
  theglobaltimehit = preStepPoint->GetGlobalTime() / nanosecond;
  incidentEnergy = (aStep->GetPreStepPoint()->GetTotalEnergy() / eV);
  tSlice = (postStepPoint->GetGlobalTime()) / nanosecond;
  tSliceID = (int)tSlice;
  unitID = setDetUnitId(aStep);

  if (verbosity_)
    LogDebug("PPSSimDiamond") << "UNITa " << unitID << "\n";

  primaryID = theTrack->GetTrackID();
  Pabs = (aStep->GetPreStepPoint()->GetMomentum().mag()) / GeV;
  p_x = (aStep->GetPreStepPoint()->GetMomentum().x()) / GeV;
  p_y = (aStep->GetPreStepPoint()->GetMomentum().y()) / GeV;
  p_z = (aStep->GetPreStepPoint()->GetMomentum().z()) / GeV;
  Tof = aStep->GetPreStepPoint()->GetGlobalTime() / nanosecond;
  Eloss = (aStep->GetPreStepPoint()->GetTotalEnergy() / eV);  //pps added
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();

  //corrected phi and theta treatment
  G4ThreeVector gmd = aStep->GetPreStepPoint()->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd = ((G4TouchableHistory*)(aStep->GetPreStepPoint()->GetTouchable()))
                          ->GetHistory()
                          ->GetTopTransform()
                          .TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  ThetaAtEntry = lnmd.theta();
  PhiAtEntry = lnmd.phi();

  if (IsPrimary(theTrack))
    ParentId = 0;
  else
    ParentId = theTrack->GetParentID();

  Vx = theTrack->GetVertexPosition().x() / mm;
  Vy = theTrack->GetVertexPosition().y() / mm;
  Vz = theTrack->GetVertexPosition().z() / mm;
}

uint32_t PPSDiamondSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->GetUnitID(aStep));
}

void PPSDiamondSD::StoreHit(PPSDiamondG4Hit* hit) {
  if (hit == nullptr) {
    if (verbosity_)
      LogDebug("PPSSimDiamond") << "PPSDiamond: hit to be stored is NULL !!"
                                << "\n";
    return;
  }

  theHC->insert(hit);
}

void PPSDiamondSD::ImportInfotoHit() {
  currentHit = new PPSDiamondG4Hit;
  currentHit->setTrackID(primaryID);
  currentHit->setTimeSlice(tSlice);
  currentHit->setUnitID(unitID);
  currentHit->setIncidentEnergy(incidentEnergy);
  currentHit->setPabs(Pabs);
  currentHit->setTof(Tof);
  currentHit->setEnergyLoss(Eloss);
  currentHit->setParticleType(ParticleType);
  currentHit->setThetaAtEntry(ThetaAtEntry);
  currentHit->setPhiAtEntry(PhiAtEntry);
  currentHit->setEntry(hitPoint);
  currentHit->setExit(exitPoint);
  currentHit->setLocalEntry(theLocalEntryPoint);
  currentHit->setLocalExit(theLocalExitPoint);
  currentHit->setParentId(ParentId);
  currentHit->setVx(Vx);
  currentHit->setVy(Vy);
  currentHit->setVz(Vz);
  currentHit->set_p_x(p_x);
  currentHit->set_p_y(p_y);
  currentHit->set_p_z(p_z);
  currentHit->setGlobalTimehit(Globaltimehit);

  StoreHit(currentHit);
  LogDebug("PPSSimDiamond") << "STORED HIT IN: " << unitID << "\n";
}

G4ThreeVector PPSDiamondSD::SetToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);

  return localPoint;
}

void PPSDiamondSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  for (int j = 0; j < theHC->entries() && j < 15000; j++) {
    PPSDiamondG4Hit* aHit = (*theHC)[j];

    Local3DPoint Entrata(aHit->getLocalEntry().x(), aHit->getLocalEntry().y(), aHit->getLocalEntry().z());
    Local3DPoint Uscita(aHit->getLocalExit().x(), aHit->getLocalExit().y(), aHit->getLocalExit().z());
    slave->processHits(PSimHit(Entrata,
                               Uscita,
                               aHit->getPabs(),
                               aHit->getTof(),
                               aHit->getEnergyLoss(),
                               aHit->getParticleType(),
                               aHit->getUnitID(),
                               aHit->getTrackID(),
                               aHit->getThetaAtEntry(),
                               aHit->getPhiAtEntry()));
  }
  Summarize();
}

void PPSDiamondSD::Summarize() {}

void PPSDiamondSD::clear() {}

void PPSDiamondSD::DrawAll() {}

void PPSDiamondSD::PrintAll() {
  LogDebug("PPSSimDiamond") << "PPSDiamond: Collection " << theHC->GetName() << "\n";
  theHC->PrintAllHits();
}

void PPSDiamondSD::fillHits(edm::PSimHitContainer& c, const std::string& n) {
  if (slave->name() == n)
    c = slave->hits();
}

void PPSDiamondSD::SetNumberingScheme(PPSVDetectorOrganization* scheme) {
  if (numberingScheme)
    delete numberingScheme;
  numberingScheme = scheme;
  LogDebug("PPSSimDiamond") << "SetNumberingScheme " << numberingScheme << "\n";
}

void PPSDiamondSD::update(const BeginOfEvent* i) {
  LogDebug("PPSSimDiamond") << " Dispatched BeginOfEvent !"
                            << "\n";
  clearHits();
  eventno = (*i)()->GetEventID();
}

void PPSDiamondSD::update(const ::EndOfEvent*) {}

void PPSDiamondSD::clearTrack(G4Track* track) { track->SetTrackStatus(fStopAndKill); }

void PPSDiamondSD::clearHits() { slave->Initialize(); }

bool PPSDiamondSD::IsPrimary(const G4Track* track) {
  TrackInformation* info = dynamic_cast<TrackInformation*>(track->GetUserInformation());
  return info && info->isPrimary();
}
