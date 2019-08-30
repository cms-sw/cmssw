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
    : SensitiveTkDetector(name_, es, clg, p),
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
  verbn = 10000;
  SetVerboseLevel(verbn);
  slave_ = new TrackingSlaveSD(name_);
  if (name_ == "CTPPSPixelHits") {
    numberingScheme_ = dynamic_cast<PPSVDetectorOrganization*>(new PPSPixelNumberingScheme());
  } else {
    edm::LogWarning("PPSSim") << "PPSPixelSD: ReadoutName not supported\n";
  }

  edm::LogInfo("PPSSim") << "PPSPixelSD: Instantiation completed";
}

PPSPixelSD::~PPSPixelSD() {
  delete slave_;
  delete numberingScheme_;
}

bool PPSPixelSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (!aStep)
    return true;

  GetStepInfo(aStep);
  if (!HitExists() && edeposit_ > 0.)
    CreateNewHit();
  else if (!HitExists() && ((unitID_ == 1111 || unitID_ == 2222) && ParentId_ == 0 && ParticleType_ == 2212))
    CreateNewHitEvo();
  return true;
}

uint32_t PPSPixelSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme_ == nullptr ? 0 : numberingScheme_->GetUnitID(aStep));
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
  for (int j = 0; j < theHC_->entries() && j < 15000; j++) {
    PPSPixelG4Hit* aHit = (*theHC_)[j];
#ifdef debug
    LogDebug("PPSSim") << "HIT NUMERO " << j << "unit ID = " << aHit->getUnitID() << "\n"
                       << "               "
                       << "enrty z " << aHit->getEntry().z() << "\n"
                       << "               "
                       << "theta   " << aHit->getThetaAtEntry() << "\n";
#endif

    Local3DPoint Enter(aHit->getEntryPoint().x(), aHit->getEntryPoint().y(), aHit->getEntryPoint().z());
    Local3DPoint Exit(aHit->getExitPoint().x(), aHit->getExitPoint().y(), aHit->getExitPoint().z());
    slave_->processHits(PSimHit(Enter,
                                Exit,
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

G4ThreeVector PPSPixelSD::SetToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint_->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return localPoint;
}

void PPSPixelSD::GetStepInfo(const G4Step* aStep) {
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

  Posizio_ = hitPoint_;
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

bool PPSPixelSD::HitExists() {
  if (primaryID_ < 1) {
    edm::LogWarning("PPSSim") << "***** PPSPixelSD error: primaryID = " << primaryID_ << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track
  if (tSliceID_ == tsID_ && unitID_ == previousUnitID_) {
    UpdateHit();
    return true;
  }

  // Reset entry point for new primary
  if (primaryID_ != primID_)
    ResetForNewPrimary();

  //look in the HitContainer whether a hit with the same primID_, unitID_,
  //tSliceID_ already exists:
  bool found = false;

  for (int j = 0; j < theHC_->entries() && !found; j++) {
    PPSPixelG4Hit* aPreviousHit = (*theHC_)[j];
    if (aPreviousHit->getTrackID() == primaryID_ && aPreviousHit->getTimeSliceID() == tSliceID_ &&
        aPreviousHit->getUnitID() == unitID_) {
      currentHit_ = aPreviousHit;
      found = true;
    }
  }

  if (found) {
    UpdateHit();
    return true;
  }
  return false;
}

void PPSPixelSD::CreateNewHit() {
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

  currentHit_->setPabs(Pabs_);
  currentHit_->setTof(Tof_);
  currentHit_->setEnergyLoss(Eloss_);
  currentHit_->setParticleType(ParticleType_);
  currentHit_->setThetaAtEntry(ThetaAtEntry_);
  currentHit_->setPhiAtEntry(PhiAtEntry_);

  currentHit_->setMeanPosition(Posizio_);
  currentHit_->setEntryPoint(theEntryPoint_);
  currentHit_->setExitPoint(theExitPoint_);

  currentHit_->setParentId(ParentId_);
  currentHit_->setVx(Vx_);
  currentHit_->setVy(Vy_);
  currentHit_->setVz(Vz_);

  UpdateHit();

  StoreHit(currentHit_);
}

void PPSPixelSD::CreateNewHitEvo() {
  currentHit_ = new PPSPixelG4Hit;
  currentHit_->setTrackID(primaryID_);
  currentHit_->setTimeSlice(tSlice_);
  currentHit_->setUnitID(unitID_);
  currentHit_->setIncidentEnergy(incidentEnergy_);

  currentHit_->setPabs(Pabs_);
  currentHit_->setTof(Tof_);
  currentHit_->setEnergyLoss(Eloss_);
  currentHit_->setParticleType(ParticleType_);
  currentHit_->setThetaAtEntry(ThetaAtEntry_);
  currentHit_->setPhiAtEntry(PhiAtEntry_);

  currentHit_->setEntryPoint(theEntryPoint_);
  currentHit_->setExitPoint(theExitPoint_);

  currentHit_->setParentId(ParentId_);
  currentHit_->setVx(Vx_);
  currentHit_->setVy(Vy_);
  currentHit_->setVz(Vz_);

  G4ThreeVector _PosizioEvo;
  int flagAcc = 0;
  _PosizioEvo = PosizioEvo(Posizio_, Vx_, Vy_, Vz_, Pabs_, flagAcc);

  if (flagAcc == 1) {
    currentHit_->setMeanPosition(_PosizioEvo);

    UpdateHit();

    StoreHit(currentHit_);
  }
}

G4ThreeVector PPSPixelSD::PosizioEvo(
    const G4ThreeVector& Pos, double vx, double vy, double vz, double pabs, int& accettanza) {
  accettanza = 0;
  //Pos.xyz() in mm
  G4ThreeVector PosEvo;
  double ThetaX = atan((Pos.x() - vx) / (Pos.z() - vz));
  double ThetaY = atan((Pos.y() - vy) / (Pos.z() - vz));
  double X_at_0 = (vx - ((Pos.x() - vx) / (Pos.z() - vz)) * vz) / 1000.;
  double Y_at_0 = (vy - ((Pos.y() - vy) / (Pos.z() - vz)) * vz) / 1000.;

  double csi = fabs((7000. - pabs) / 7000.);

  // all in m
  const int no_rp = 4;
  double x_par[no_rp + 1];
  double y_par[no_rp + 1];
  //rp z position
  double rp[no_rp] = {141., 149., 198., 220.};
  //{lx0,mlx} for each rp; Lx=lx0+mlx*csi
  double leffx[][2] = {{122.5429, -46.9312}, {125.4194, -49.1849}, {152.6, -81.157}, {98.8914, -131.8390}};
  //{ly0,mly} for each rp; Ly=ly0+mly*csi
  double leffy[][2] = {{124.2314, -55.4852}, {127.7825, -57.4503}, {179.455, -76.274}, {273.0931, -40.4626}};
  //{vx0,mvx0} for each rp; vx=vx0+mvx*csi
  double avx[][2] = {{0.515483, -1.0123}, {0.494122, -1.0534}, {0.2217, -1.483}, {0.004633, -1.0719}};
  //{vy0,mvy0} for each rp; vy=vy0+mvy*csi
  double avy[][2] = {{0.371418, -1.6327}, {0.349035, -1.6955}, {0.0815, -2.59}, {0.007592, -4.0841}};
  //{D0,md,a,b} for each rp; D=D0+(md+a*thetax)*csi+b*thetax
  double ddx[][4] = {{-0.082336, -0.092513, 112.3436, -82.5029},
                     {-0.086927, -0.097670, 114.9513, -82.9835},
                     {-0.092117, -0.0915, 180.6236, -82.443},
                     {-0.050470, 0.058837, 208.1106, 20.8198}};
  // {10sigma_x+0.5mm,10sigma_y+0.5mm}
  double detlim[][2] = {{0, 0}, {0.0028, 0.0021}, {0, 0}, {0.0008, 0.0013}};
  //{rmax,dmax}
  double pipelim[][2] = {{0.026, 0.026}, {0.04, 0.04}, {0.0226, 0.0177}, {0.04, 0.04}};

  for (int j = 0; j < no_rp; j++) {
    y_par[j] = ThetaY * (leffy[j][0] + leffy[j][1] * csi) + (avy[j][0] + avy[j][1] * csi) * Y_at_0;
    x_par[j] = ThetaX * (leffx[j][0] + leffx[j][1] * csi) + (avx[j][0] + avx[j][1] * csi) * X_at_0 -
               csi * (ddx[j][0] + (ddx[j][1] + ddx[j][2] * ThetaX) * csi + ddx[j][3] * ThetaX);
  }

  //pass TAN@141
  if (fabs(y_par[0]) < pipelim[0][1] && sqrt((y_par[0] * y_par[0]) + (x_par[0] * x_par[0])) < pipelim[0][0]) {
    //pass 149
    if ((sqrt((y_par[1] * y_par[1]) + (x_par[1] * x_par[1])) < pipelim[1][0]) &&
        (fabs(y_par[1]) > detlim[1][1] || x_par[1] > detlim[1][0])) {
      accettanza = 1;
    }
  }

  //pass TAN@141
  if (fabs(y_par[0]) < pipelim[0][1] && sqrt((y_par[0]) * (y_par[0]) + (x_par[0]) * (x_par[0])) < pipelim[0][0]) {
    //pass Q5@198
    if (fabs(y_par[2]) < pipelim[2][1] && sqrt((y_par[2] * y_par[2]) + (x_par[2] * x_par[2])) < pipelim[2][0]) {
      //pass 220
      if ((sqrt((y_par[3] * y_par[3]) + (x_par[3] * x_par[3])) < pipelim[3][0]) &&
          (fabs(y_par[3]) > detlim[3][1] || x_par[3] > detlim[3][0])) {
        accettanza = 1;

        PosEvo.setX(1000 * x_par[3]);
        PosEvo.setY(1000 * y_par[3]);
        PosEvo.setZ(1000 * rp[3]);
        if (Pos.z() < vz)
          PosEvo.setZ(-1000 * rp[3]);
      }
    }
  }
  return PosEvo;
}

void PPSPixelSD::UpdateHit() {
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

void PPSPixelSD::StoreHit(PPSPixelG4Hit* hit) {
  if (primID_ < 0)
    return;
  if (hit == nullptr) {
    edm::LogWarning("PPSSim") << "PPSPixelSD: hit to be stored is NULL !!";
    return;
  }

  theHC_->insert(hit);
}

void PPSPixelSD::ResetForNewPrimary() {
  entrancePoint_ = SetToLocal(hitPoint_);

  incidentEnergy_ = preStepPoint_->GetKineticEnergy();
}

void PPSPixelSD::Summarize() {}
