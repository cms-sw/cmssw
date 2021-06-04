// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemSD
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "SimG4CMS/Forward/interface/TotemSD.h"
#include "SimG4CMS/Forward/interface/TotemNumberMerger.h"
#include "SimG4CMS/Forward/interface/TotemT1NumberingScheme.h"
#include "SimG4CMS/Forward/interface/TotemT2NumberingSchemeGem.h"
#include "SimG4CMS/Forward/interface/TotemRPNumberingScheme.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//
// constructors and destructor
//
TotemSD::TotemSD(const std::string& name,
                 const edm::EventSetup& es,
                 const SensitiveDetectorCatalog& clg,
                 edm::ParameterSet const& p,
                 const SimTrackManager* manager)
    : SensitiveTkDetector(name, clg),
      numberingScheme(nullptr),
      hcID(-1),
      theHC(nullptr),
      theManager(manager),
      currentHit(nullptr),
      theTrack(nullptr),
      currentPV(nullptr),
      unitID(0),
      previousUnitID(0),
      preStepPoint(nullptr),
      postStepPoint(nullptr) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("TotemSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  slave = new TrackingSlaveSD(name);

  if (name == "TotemHitsT1") {
    numberingScheme = dynamic_cast<TotemVDetectorOrganization*>(new TotemT1NumberingScheme(1));
  } else if (name == "TotemHitsT2Si") {
    numberingScheme = dynamic_cast<TotemVDetectorOrganization*>(new TotemT2NumberingSchemeGem(3));
  } else if (name == "TotemHitsT2Gem") {
    numberingScheme = dynamic_cast<TotemVDetectorOrganization*>(new TotemT2NumberingSchemeGem(4));
  } else if (name == "TotemHitsRP") {
    numberingScheme = dynamic_cast<TotemVDetectorOrganization*>(new TotemRPNumberingScheme(3));
  } else {
    edm::LogWarning("ForwardSim") << "TotemSD: ReadoutName not supported\n";
  }
}

TotemSD::~TotemSD() {
  delete slave;
  delete numberingScheme;
}

bool TotemSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  getStepInfo(aStep);
  if (!hitExists() && edeposit > 0.) {
    createNewHit();
    return true;
  }
  if (!hitExists() && (((unitID == 1111 || unitID == 2222) && ParentId == 0 && ParticleType == 2212))) {
    createNewHitEvo();
  }
  return true;
}

uint32_t TotemSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}

void TotemSD::Initialize(G4HCofThisEvent* HCE) {
  LogDebug("ForwardSim") << "TotemSD : Initialize called for " << GetName();

  theHC = new TotemG4HitCollection(GetName(), collectionName[0]);
  if (hcID < 0)
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);

  tsID = -2;
  primID = -2;
}

void TotemSD::EndOfEvent(G4HCofThisEvent*) {
  // here we loop over transient hits and make them persistent
  int thehc_entries = theHC->entries();
  for (int j = 0; j < thehc_entries && j < 15000; j++) {
    TotemG4Hit* aHit = (*theHC)[j];
#ifdef ddebug
    LogDebug("ForwardSim") << "HIT NUMERO " << j << "unit ID = " << aHit->getUnitID() << "\n"
                           << "               "
                           << "enrty z " << aHit->getEntry().z() << "\n"
                           << "               "
                           << "theta   " << aHit->getThetaAtEntry() << "\n";
#endif
    Local3DPoint theExitPoint(0, 0, 0);
    Local3DPoint Entrata(aHit->getEntry().x(), aHit->getEntry().y(), aHit->getEntry().z());
    slave->processHits(PSimHit(Entrata,
                               theExitPoint,
                               aHit->getPabs(),
                               aHit->getTof(),
                               aHit->getEnergyLoss(),
                               aHit->getParticleType(),
                               aHit->getUnitID(),
                               aHit->getTrackID(),
                               aHit->getThetaAtEntry(),
                               aHit->getPhiAtEntry()));
  }
}

void TotemSD::PrintAll() {
  LogDebug("ForwardSim") << "TotemSD: Collection " << theHC->GetName();
  theHC->PrintAllHits();
}

void TotemSD::fillHits(edm::PSimHitContainer& cc, const std::string& hname) {
  if (slave->name() == hname) {
    cc = slave->hits();
  }
}

void TotemSD::update(const BeginOfEvent* i) {
  LogDebug("ForwardSim") << " Dispatched BeginOfEvent for " << GetName() << " !";
  clearHits();
}

void TotemSD::clearHits() { slave->Initialize(); }

G4ThreeVector TotemSD::setToLocal(const G4ThreeVector& global) {
  G4ThreeVector localPoint;
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return localPoint;
}

void TotemSD::getStepInfo(const G4Step* aStep) {
  preStepPoint = aStep->GetPreStepPoint();
  postStepPoint = aStep->GetPostStepPoint();
  theTrack = aStep->GetTrack();
  hitPoint = preStepPoint->GetPosition();
  currentPV = preStepPoint->GetPhysicalVolume();

  // double weight = 1;
  G4String name = currentPV->GetName();
  name.assign(name, 0, 4);
  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  edeposit = aStep->GetTotalEnergyDeposit();

  tSlice = (postStepPoint->GetGlobalTime()) / nanosecond;
  tSliceID = (int)tSlice;
  unitID = setDetUnitId(aStep);
#ifdef debug
  LogDebug("ForwardSim") << "UNITa " << unitID;
#endif
  primaryID = theTrack->GetTrackID();

  Posizio = hitPoint;
  Pabs = aStep->GetPreStepPoint()->GetMomentum().mag() / GeV;
  Tof = aStep->GetPostStepPoint()->GetGlobalTime() / nanosecond;

  Eloss = aStep->GetTotalEnergyDeposit() / GeV;
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();

  ThetaAtEntry = aStep->GetPreStepPoint()->GetPosition().theta() / deg;
  PhiAtEntry = aStep->GetPreStepPoint()->GetPosition().phi() / deg;

  ParentId = theTrack->GetParentID();
  Vx = theTrack->GetVertexPosition().x();
  Vy = theTrack->GetVertexPosition().y();
  Vz = theTrack->GetVertexPosition().z();
}

bool TotemSD::hitExists() {
  if (primaryID < 1) {
    edm::LogWarning("ForwardSim") << "***** TotemSD error: primaryID = " << primaryID << " maybe detector name changed";
  }

  // Update if in the same detector, time-slice and for same track
  //  if (primaryID == primID && tSliceID == tsID && unitID==previousUnitID) {
  if (tSliceID == tsID && unitID == previousUnitID) {
    updateHit();
    return true;
  }

  // Reset entry point for new primary
  if (primaryID != primID)
    resetForNewPrimary();

  //look in the HitContainer whether a hit with the same primID, unitID,
  //tSliceID already exists:

  bool found = false;
  int thehc_entries = theHC->entries();
  for (int j = 0; j < thehc_entries && !found; j++) {
    TotemG4Hit* aPreviousHit = (*theHC)[j];
    if (aPreviousHit->getTrackID() == primaryID && aPreviousHit->getTimeSliceID() == tSliceID &&
        aPreviousHit->getUnitID() == unitID) {
      currentHit = aPreviousHit;
      found = true;
      break;
    }
  }

  if (found) {
    updateHit();
    return true;
  } else {
    return false;
  }
}

void TotemSD::createNewHit() {
#ifdef debug
  LogDebug("ForwardSim") << "TotemSD CreateNewHit for"
                         << " PV " << currentPV->GetName() << " PVid = " << currentPV->GetCopyNo()
                         << " MVid = " << currentPV->GetMother()->GetCopyNo() << " Unit " << unitID << "\n"
                         << " primary " << primaryID << " time slice " << tSliceID << " For Track  "
                         << theTrack->GetTrackID() << " which is a " << theTrack->GetDefinition()->GetParticleName();

  if (theTrack->GetTrackID() == 1) {
    LogDebug("ForwardSim") << " of energy " << theTrack->GetTotalEnergy();
  } else {
    LogDebug("ForwardSim") << " daughter of part. " << theTrack->GetParentID();
  }

  if (theTrack->GetCreatorProcess() != nullptr)
    LogDebug("ForwardSim") << theTrack->GetCreatorProcess()->GetProcessName();
  else
    LogDebug("ForwardSim") << "NO process";
#endif

  currentHit = new TotemG4Hit;
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

  currentHit->setEntry(Posizio.x(), Posizio.y(), Posizio.z());

  currentHit->setParentId(ParentId);
  currentHit->setVx(Vx);
  currentHit->setVy(Vy);
  currentHit->setVz(Vz);

  updateHit();

  storeHit(currentHit);
}

void TotemSD::createNewHitEvo() {
  // LogDebug("ForwardSim") << "INSIDE CREATE NEW HIT EVO ";

  currentHit = new TotemG4Hit;
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

  //  LogDebug("ForwardSim") << Posizio.x() << " " << Posizio.y() << " " << Posizio.z();

  currentHit->setParentId(ParentId);
  currentHit->setVx(Vx);
  currentHit->setVy(Vy);
  currentHit->setVz(Vz);

  G4ThreeVector _PosizioEvo;
  int flagAcc = 0;
  _PosizioEvo = posizioEvo(Posizio, Vx, Vy, Vz, Pabs, flagAcc);

  if (flagAcc == 1) {
    currentHit->setEntry(_PosizioEvo.x(), _PosizioEvo.y(), _PosizioEvo.z());

    updateHit();

    storeHit(currentHit);
  }
  // LogDebug("ForwardSim") << "STORED HIT IN: " << unitID;
}

G4ThreeVector TotemSD::posizioEvo(
    const G4ThreeVector& Pos, double vx, double vy, double vz, double pabs, int& accettanza) {
  accettanza = 0;
  //Pos.xyz() in mm
  G4ThreeVector PosEvo;
  double ThetaX = std::atan((Pos.x() - vx) / (Pos.z() - vz));
  double ThetaY = std::atan((Pos.y() - vy) / (Pos.z() - vz));
  double X_at_0 = (vx - ((Pos.x() - vx) / (Pos.z() - vz)) * vz) / 1000.;
  double Y_at_0 = (vy - ((Pos.y() - vy) / (Pos.z() - vz)) * vz) / 1000.;

  //  double temp_evo_X;
  //  double temp_evo_Y;
  //  double temp_evo_Z;
  //  temp_evo_Z = std::abs(Pos.z())/Pos.z()*220000.;

  //csi=-dp/d
  double csi = std::abs((7000. - pabs) / 7000.);

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
    //y=Ly*thetay+vy*y0
    //x=Lx*thetax+vx*x0-csi*D
    y_par[j] = ThetaY * (leffy[j][0] + leffy[j][1] * csi) + (avy[j][0] + avy[j][1] * csi) * Y_at_0;
    x_par[j] = ThetaX * (leffx[j][0] + leffx[j][1] * csi) + (avx[j][0] + avx[j][1] * csi) * X_at_0 -
               csi * (ddx[j][0] + (ddx[j][1] + ddx[j][2] * ThetaX) * csi + ddx[j][3] * ThetaX);
  }

  //pass TAN@141
  if (std::abs(y_par[0]) < pipelim[0][1] && sqrt((y_par[0] * y_par[0]) + (x_par[0] * x_par[0])) < pipelim[0][0]) {
    //pass 149
    if ((sqrt((y_par[1] * y_par[1]) + (x_par[1] * x_par[1])) < pipelim[1][0]) &&
        (std::abs(y_par[1]) > detlim[1][1] || x_par[1] > detlim[1][0])) {
      accettanza = 1;
    }
  }

  //pass TAN@141
  if (std::abs(y_par[0]) < pipelim[0][1] && sqrt((y_par[0]) * (y_par[0]) + (x_par[0]) * (x_par[0])) < pipelim[0][0]) {
    //pass Q5@198
    if (std::abs(y_par[2]) < pipelim[2][1] && sqrt((y_par[2] * y_par[2]) + (x_par[2] * x_par[2])) < pipelim[2][0]) {
      //pass 220
      if ((sqrt((y_par[3] * y_par[3]) + (x_par[3] * x_par[3])) < pipelim[3][0]) &&
          (std::abs(y_par[3]) > detlim[3][1] || x_par[3] > detlim[3][0])) {
        accettanza = 1;

        PosEvo.setX(1000 * x_par[3]);
        PosEvo.setY(1000 * y_par[3]);
        PosEvo.setZ(1000 * rp[3]);
        if (Pos.z() < vz)
          PosEvo.setZ(-1000 * rp[3]);
      }
    }
  }
  /*
  LogDebug("ForwardSim") << "\n"
			 << "ACCETTANZA: "<<accettanza << "\n" 
			 << "CSI: "<< csi << "\n"
			 << "Theta_X: " << ThetaX << "\n"
			 << "Theta_Y: " << ThetaY << "\n"
			 << "X_at_0: "<< X_at_0 << "\n"
			 << "Y_at_0: "<< Y_at_0 << "\n" 
			 << "x_par[3]: "<< x_par[3] << "\n"
			 << "y_par[3]: "<< y_par[3] << "\n"
			 << "pos " << Pos.x() << " " << Pos.y() << " " 
			 << Pos.z() << "\n" << "V "<< vx << " " << vy << " "
			 << vz << "\n"
*/
  // --------------
  return PosEvo;
}

void TotemSD::updateHit() {
  //
  if (Eloss > 0.) {
#ifdef debug
    LogDebug("ForwardSim") << "G4TotemT1SD updateHit: add eloss " << Eloss << "\nCurrentHit=" << currentHit
                           << ", PostStepPoint=" << postStepPoint->GetPosition();
#endif

    currentHit->setEnergyLoss(Eloss);
  }

  // buffer for next steps:
  tsID = tSliceID;
  primID = primaryID;
  previousUnitID = unitID;
}

void TotemSD::storeHit(TotemG4Hit* hit) {
  if (primID < 0)
    return;
  if (hit == nullptr) {
    edm::LogWarning("ForwardSim") << "TotemSD: hit to be stored is NULL !!";
    return;
  }

  theHC->insert(hit);
}

void TotemSD::resetForNewPrimary() {
  entrancePoint = setToLocal(hitPoint);
  incidentEnergy = preStepPoint->GetKineticEnergy();
}
