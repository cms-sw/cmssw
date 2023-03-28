///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.cc
// Description: Sensitive Detector class for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/FastMath.h"

#include "SimG4CMS/Calo/interface/HGCSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"
#include "G4Trap.hh"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

//#define EDM_ML_DEBUG
//#define plotDebug

HGCSD::HGCSD(const std::string& name,
             const HGCalDDDConstants* hgc,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : CaloSD(name,
             clg,
             p,
             manager,
             (float)(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")),
      hgcons_(hgc),
      slopeMin_(0),
      levelT_(99),
      tree_(nullptr) {
  numberingScheme_.reset(nullptr);
  mouseBite_.reset(nullptr);

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit_ = m_HGC.getParameter<double>("EminHit") * CLHEP::MeV;
  storeAllG4Hits_ = m_HGC.getParameter<bool>("StoreAllG4Hits");
  rejectMB_ = m_HGC.getParameter<bool>("RejectMouseBite");
  waferRot_ = m_HGC.getParameter<bool>("RotatedWafer");
  angles_ = m_HGC.getUntrackedParameter<std::vector<double>>("WaferAngles");
  double waferSize = m_HGC.getUntrackedParameter<double>("WaferSize") * CLHEP::mm;
  double mouseBite = m_HGC.getUntrackedParameter<double>("MouseBite") * CLHEP::mm;
  mouseBiteCut_ = waferSize * tan(30.0 * CLHEP::deg) - mouseBite;

  if (storeAllG4Hits_) {
    setUseMap(false);
    setNumberCheckedHits(0);
  }
  //this is defined in the hgcsens.xml
  G4String myName = name;
  myFwdSubdet_ = ForwardSubdetector::ForwardEmpty;
  nameX_ = "HGCal";
  if (myName.find("HitsEE") != std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCEE;
    nameX_ = "HGCalEESensitive";
  } else if (myName.find("HitsHEfront") != std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEF;
    nameX_ = "HGCalHESiliconSensitive";
  } else if (myName.find("HitsHEback") != std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEB;
    nameX_ = "HGCalHEScintillatorSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "**************************************************"
                             << "\n"
                             << "*                                                *"
                             << "\n"
                             << "* Constructing a HGCSD  with name " << name << "\n"
                             << "*                                                *"
                             << "\n"
                             << "**************************************************";
#endif
  edm::LogVerbatim("HGCSim") << "HGCSD:: Threshold for storing hits: " << eminHit << " for " << nameX_ << " subdet "
                             << myFwdSubdet_;
  edm::LogVerbatim("HGCSim") << "Flag for storing individual Geant4 Hits " << storeAllG4Hits_;
  edm::LogVerbatim("HGCSim") << "Reject MosueBite Flag: " << rejectMB_ << " Size of wafer " << waferSize
                             << " Mouse Bite " << mouseBite << ":" << mouseBiteCut_ << " along " << angles_.size()
                             << " axes";
}

double HGCSD::getEnergyDeposit(const G4Step* aStep) {
  double r = aStep->GetPreStepPoint()->GetPosition().perp();
  double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());

#ifdef EDM_ML_DEBUG
  G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4LogicalVolume* lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  edm::LogVerbatim("HGCSim") << "HGCSD: Hit from standard path from " << lv->GetName() << " for Track "
                             << aStep->GetTrack()->GetTrackID() << " ("
                             << aStep->GetTrack()->GetDefinition()->GetParticleName() << ":" << parCode << ") R = " << r
                             << " Z = " << z << " slope = " << r / z << ":" << slopeMin_;
#endif

  // Apply fiductial volume
  if (r < z * slopeMin_) {
    return 0.0;
  }

  double wt1 = getResponseWt(aStep->GetTrack());
  double wt2 = aStep->GetTrack()->GetWeight();
  double destep = wt1 * aStep->GetTotalEnergyDeposit();
  if (wt2 > 0)
    destep *= wt2;

#ifdef plotDebug
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4double tmptrackE = aStep->GetTrack()->GetKineticEnergy();
  G4int parCodex = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4double angle = (aStep->GetTrack()->GetMomentumDirection().theta()) / CLHEP::deg;
  G4int layer = ((touch->GetHistoryDepth() == levelT_) ? touch->GetReplicaNumber(0) : touch->GetReplicaNumber(2));
  G4int ilayer = (layer - 1) / 3;
  if (aStep->GetTotalEnergyDeposit() > 0) {
    t_Layer_.emplace_back(ilayer);
    t_Parcode_.emplace_back(parCodex);
    t_dEStep1_.emplace_back(aStep->GetTotalEnergyDeposit());
    t_dEStep2_.emplace_back(destep);
    t_TrackE_.emplace_back(tmptrackE);
    t_Angle_.emplace_back(angle);
  }
#endif

  return destep;
}

uint32_t HGCSD::setDetUnitId(const G4Step* aStep) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  //determine the exact position in global coordinates in the mass geometry
  G4ThreeVector hitPoint = preStepPoint->GetPosition();
  float globalZ = touch->GetTranslation(0).z();
  int iz(globalZ > 0 ? 1 : -1);

  //convert to local coordinates (=local to the current volume):
  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

  //get the det unit id with
  ForwardSubdetector subdet = myFwdSubdet_;

  int layer(-1), moduleLev(-1), module(-1), cell(-1);
  if (touch->GetHistoryDepth() == levelT_) {
    layer = touch->GetReplicaNumber(0);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth() << " name " << touch->GetVolume(0)->GetName()
                               << " layer:module:cell " << layer << ":" << moduleLev << ":" << module << ":" << cell;
#endif
  } else {
    layer = touch->GetReplicaNumber(2);
    module = touch->GetReplicaNumber(1);
    cell = touch->GetReplicaNumber(0);
    moduleLev = 1;
  }
#ifdef EDM_ML_DEBUG
  const G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
  edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth() << " name " << touch->GetVolume(0)->GetName()
                             << ":" << touch->GetReplicaNumber(0) << "   " << touch->GetVolume(1)->GetName() << ":"
                             << touch->GetReplicaNumber(1) << "   " << touch->GetVolume(2)->GetName() << ":"
                             << touch->GetReplicaNumber(2) << "    layer:module:cell " << layer << ":" << moduleLev
                             << ":" << module << ":" << cell << " Material " << mat->GetName() << ":"
                             << mat->GetRadlen();
  for (int k = 0; k < touch->GetHistoryDepth(); ++k)
    edm::LogVerbatim("HGCSim") << "Level [" << k << "] " << touch->GetVolume(k)->GetName() << ":"
                               << touch->GetReplicaNumber(k);
#endif
  // The following statement should be examined later before elimination
  // VI: this is likely a check if media is vacuum - not needed
  if (aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() > 100000.)
    return 0;

  uint32_t id = setDetUnitId(subdet, layer, module, cell, iz, localpos);
  if (rejectMB_ && id != 0) {
    int det, z, lay, wafer, type, ic;
    HGCalTestNumbering::unpackHexagonIndex(id, det, z, lay, wafer, type, ic);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "ID " << std::hex << id << std::dec << " Input " << subdet << ":" << layer << ":"
                               << module << ":" << cell << ":" << iz << localpos.x() << ":" << localpos.y()
                               << " Decode " << det << ":" << z << ":" << lay << ":" << wafer << ":" << type << ":"
                               << ic;
#endif
    G4ThreeVector local =
        ((moduleLev >= 0) ? (touch->GetHistory()->GetTransform(moduleLev).TransformPoint(hitPoint)) : G4ThreeVector());
    if (mouseBite_->exclude(local, z, layer, wafer, 0))
      id = 0;
  }
  return id;
}

void HGCSD::update(const BeginOfJob* job) {
  if (hgcons_ != nullptr) {
    geom_mode_ = hgcons_->geomMode();
    slopeMin_ = hgcons_->minSlope();
    levelT_ = hgcons_->levelTop();
    numberingScheme_ = std::make_unique<HGCNumberingScheme>(*hgcons_, nameX_);
    if (rejectMB_)
      mouseBite_ = std::make_unique<HGCMouseBite>(*hgcons_, angles_, mouseBiteCut_, waferRot_);
  } else {
    edm::LogError("HGCSim") << "HGCSD : Cannot find HGCalDDDConstants for " << nameX_;
    throw cms::Exception("Unknown", "HGCSD") << "Cannot find HGCalDDDConstants for " << nameX_ << "\n";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCSD::Initialized with mode " << geom_mode_ << " Slope cut " << slopeMin_
                             << " top Level " << levelT_;
#endif
}

void HGCSD::initRun() {
#ifdef plotDebug
  edm::Service<TFileService> tfile;
  if (tfile.isAvailable()) {
    tree_ = tfile->make<TTree>("TreeHGCSD", "TreeHGCSD");
    tree_->Branch("EventID", &t_EventID_);
    tree_->Branch("Layer", &t_Layer_);
    tree_->Branch("ParticleCode", &t_Parcode_);
    tree_->Branch("dEStepOriginal", &t_dEStep1_);
    tree_->Branch("dEStepWeighted", &t_dEStep2_);
    tree_->Branch("TrackEnergy", &t_TrackE_);
    tree_->Branch("ThetaAngle", &t_Angle_);
  }
#endif
}

void HGCSD::initEvent(const BeginOfEvent* g4Event) {
  const G4Event* evt = (*g4Event)();
  t_EventID_ = evt->GetEventID();
#ifdef plotDebug
  t_Layer_.clear();
  t_Parcode_.clear();
  t_dEStep1_.clear();
  t_dEStep2_.clear();
  t_TrackE_.clear();
  t_Angle_.clear();
#endif
}

void HGCSD::endEvent() {
#ifdef plotDebug
  if (tree_)
    tree_->Fill();
#endif
}

bool HGCSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit_));
}

uint32_t HGCSD::setDetUnitId(ForwardSubdetector& subdet, int layer, int module, int cell, int iz, G4ThreeVector& pos) {
  uint32_t id = numberingScheme_ ? numberingScheme_->getUnitID(subdet, layer, module, cell, iz, pos) : 0;
  return id;
}
