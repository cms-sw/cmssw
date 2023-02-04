///////////////////////////////////////////////////////////////////////////////
// File: HGCScintSD.cc
// Description: Sensitive Detector class for the Scintillator part of
//              High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/FastMath.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "SimG4CMS/Calo/interface/HGCScintSD.h"
#include "SimG4CMS/Calo/interface/CaloSimUtils.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
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

HGCScintSD::HGCScintSD(const std::string& name,
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
      levelT1_(99),
      levelT2_(99),
      firstLayer_(0) {
  numberingScheme_.reset(nullptr);

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCScintSD");
  eminHit_ = m_HGC.getParameter<double>("EminHit") * CLHEP::MeV;
  fiducialCut_ = m_HGC.getParameter<bool>("FiducialCut");
  distanceFromEdge_ = m_HGC.getParameter<double>("DistanceFromEdge");
  useBirk_ = m_HGC.getParameter<bool>("UseBirkLaw");
  birk1_ = m_HGC.getParameter<double>("BirkC1") * (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk2_ = m_HGC.getParameter<double>("BirkC2");
  birk3_ = m_HGC.getParameter<double>("BirkC3");
  storeAllG4Hits_ = m_HGC.getParameter<bool>("StoreAllG4Hits");
  checkID_ = m_HGC.getUntrackedParameter<bool>("CheckID");
  fileName_ = m_HGC.getUntrackedParameter<std::string>("TileFileName");
  verbose_ = m_HGC.getUntrackedParameter<int>("Verbosity");

  if (storeAllG4Hits_) {
    setUseMap(false);
    setNumberCheckedHits(0);
  }

  //this is defined in the hgcsens.xml
  G4String myName = name;
  mydet_ = DetId::Forward;
  nameX_ = "HGCal";
  if (myName.find("HitsHEback") != std::string::npos) {
    mydet_ = DetId::HGCalHSc;
    nameX_ = "HGCalHEScintillatorSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "**************************************************"
                             << "\n"
                             << "*                                                *"
                             << "\n"
                             << "* Constructing a HGCScintSD  with name " << name << "\n"
                             << "*                                                *"
                             << "\n"
                             << "**************************************************";
#endif
  edm::LogVerbatim("HGCSim") << "HGCScintSD:: Threshold for storing hits: " << eminHit_ << " for " << nameX_
                             << " detector " << mydet_ << " File " << fileName_;
  edm::LogVerbatim("HGCSim") << "Flag for storing individual Geant4 Hits " << storeAllG4Hits_;
  edm::LogVerbatim("HGCSim") << "Fiducial volume cut with cut from eta/phi "
                             << "boundary " << fiducialCut_ << " at " << distanceFromEdge_;
  edm::LogVerbatim("HGCSim") << "Use of Birks law is set to      " << useBirk_
                             << "  with three constants kB = " << birk1_ << ", C1 = " << birk2_ << ", C2 = " << birk3_;

  if (!fileName_.empty()) {
    edm::FileInPath filetmp("SimG4CMS/Calo/data/" + fileName_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCSim") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = CaloSimUtils::splitString(std::string(buffer));
        if (items.size() > 2) {
          int layer = std::atoi(items[0].c_str());
          int ring = std::atoi(items[1].c_str());
          int phi = std::atoi(items[2].c_str());
          tiles_.emplace_back(HGCalTileIndex::tileIndex(layer, ring, phi));
        }
      }
      edm::LogVerbatim("HGCSim") << "Reads in " << tiles_.size() << " tile information from " << fileName_;
      fInput.close();
    }
  }
}

double HGCScintSD::getEnergyDeposit(const G4Step* aStep) {
  double r = aStep->GetPreStepPoint()->GetPosition().perp();
  double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());
#ifdef EDM_ML_DEBUG
  G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4String parName = aStep->GetTrack()->GetDefinition()->GetParticleName();
  G4LogicalVolume* lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  edm::LogVerbatim("HGCSim") << "HGCScintSD: Hit from standard path from " << lv->GetName() << " for Track "
                             << aStep->GetTrack()->GetTrackID() << " (" << parCode << ":" << parName << ") R = " << r
                             << " Z = " << z << " slope = " << r / z << ":" << slopeMin_;
#endif
  // Apply fiducial cut
  if (r < z * slopeMin_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCScintSD: Fiducial Volume cut";
#endif
    return 0.0;
  }

  double wt1 = getResponseWt(aStep->GetTrack());
  double wt2 = aStep->GetTrack()->GetWeight();
  double wt3 = (useBirk_ ? getAttenuation(aStep, birk1_, birk2_, birk3_) : 1.0);
  double destep = weight_ * wt1 * wt3 * (aStep->GetTotalEnergyDeposit());
  if (wt2 > 0)
    destep *= wt2;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalSim") << "HGCScintSD: weights= " << weight_ << ":" << wt1 << ":" << wt2 << ":" << wt3
                               << " Total weight " << weight_ * wt1 * wt2 * wt3
                               << " deStep: " << aStep->GetTotalEnergyDeposit() << ":" << destep;
#endif
  return destep;
}

uint32_t HGCScintSD::setDetUnitId(const G4Step* aStep) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "DepthsTop: " << touch->GetHistoryDepth() << ":" << levelT1_ << ":" << levelT2_;
  printDetectorLevels(touch);
#endif
  //determine the exact position in global coordinates in the mass geometry
  G4ThreeVector hitPoint = preStepPoint->GetPosition();
  float globalZ = touch->GetTranslation(0).z();
  int iz(globalZ > 0 ? 1 : -1);

  int layer(0), module(-1), cell(-1);
  if ((geom_mode_ == HGCalGeometryMode::TrapezoidModule) || (geom_mode_ == HGCalGeometryMode::TrapezoidCassette)) {
    layer = touch->GetReplicaNumber(1);
  } else if ((touch->GetHistoryDepth() == levelT1_) || (touch->GetHistoryDepth() == levelT2_)) {
    layer = touch->GetReplicaNumber(0);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "DepthsTop: " << touch->GetHistoryDepth() << ":" << levelT1_ << ":" << levelT2_
                               << " name " << touch->GetVolume(0)->GetName() << " layer:module:cell " << layer << ":"
                               << module << ":" << cell;
#endif
  } else {
    layer = touch->GetReplicaNumber(3);
    module = touch->GetReplicaNumber(2);
    cell = touch->GetReplicaNumber(1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "DepthsInside: " << touch->GetHistoryDepth() << " name "
                               << touch->GetVolume(0)->GetName() << " layer:module:cell " << layer << ":" << module
                               << ":" << cell;
#endif
  }
#ifdef EDM_ML_DEBUG
  G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
  edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth() << " name " << touch->GetVolume(0)->GetName()
                             << ":" << touch->GetReplicaNumber(0) << "   " << touch->GetVolume(1)->GetName() << ":"
                             << touch->GetReplicaNumber(1) << "   " << touch->GetVolume(2)->GetName() << ":"
                             << touch->GetReplicaNumber(2) << "   " << touch->GetVolume(3)->GetName() << ":"
                             << touch->GetReplicaNumber(3) << "   " << touch->GetVolume(4)->GetName() << ":"
                             << touch->GetReplicaNumber(4) << "   "
                             << " layer:module:cell " << layer << ":" << module << ":" << cell << " Material "
                             << mat->GetName() << ":" << mat->GetRadlen();
#endif
  // The following statement should be examined later before elimination
  if (aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() > 100000.)
    return 0;

  uint32_t id = setDetUnitId(layer, module, cell, iz, hitPoint);
  bool debug(false);
  if (!tiles_.empty()) {
    HGCScintillatorDetId hid(id);
    int indx = HGCalTileIndex::tileIndex(firstLayer_ + hid.layer(), hid.ring(), hid.iphi());
    if (std::find(tiles_.begin(), tiles_.end(), indx) != tiles_.end())
      debug = true;
  }
  if (debug)
    edm::LogVerbatim("HGCSim") << "Layer:module:cell:iz " << layer << ":" << module << ":" << cell << ":" << iz
                               << "  Point (" << hitPoint.x() << ", " << hitPoint.y() << ", " << hitPoint.z() << ") "
                               << HGCScintillatorDetId(id);

  if (!isItinFidVolume(hitPoint)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "ID " << std::hex << id << std::dec << " " << HGCScintillatorDetId(id)
                               << " is rejected by fiducilal volume cut";
#endif
    id = 0;
  }
  if ((id != 0) && checkID_) {
    HGCScintillatorDetId hid1(id);
    std::string_view pid = ((hgcons_->cassetteShiftScintillator(hid1.layer(), hid1.iphi())) ? "HGCSim" : "HGCalSim");
    bool debug = (verbose_ > 0) ? true : false;
    auto xy = hgcons_->locateCell(HGCScintillatorDetId(id), debug);
    double dx = xy.first - (hitPoint.x() / CLHEP::cm);
    double dy = xy.second - (hitPoint.y() / CLHEP::cm);
    double diff = std::sqrt(dx * dx + dy * dy);
    constexpr double tol = 10.0;
    bool valid = hgcons_->isValidTrap(hid1.zside(), hid1.layer(), hid1.ring(), hid1.iphi());
    if ((diff > tol) || (!valid))
      pid = "HGCalError";
    edm::LogVerbatim(pid) << "CheckID " << HGCScintillatorDetId(id) << " input position: (" << hitPoint.x() / CLHEP::cm
                          << ", " << hitPoint.y() / CLHEP::cm << "); position from ID (" << xy.first << ", "
                          << xy.second << ") distance " << diff << " Valid " << valid
                          << " Rho = " << hitPoint.perp() / CLHEP::cm;
  }
  return id;
}

void HGCScintSD::update(const BeginOfJob* job) {
  if (hgcons_ != nullptr) {
    geom_mode_ = hgcons_->geomMode();
    slopeMin_ = hgcons_->minSlope();
    levelT1_ = hgcons_->levelTop(0);
    levelT2_ = hgcons_->levelTop(1);
    firstLayer_ = hgcons_->firstLayer() - 1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCScintSD::Initialized with mode " << geom_mode_ << " Slope cut " << slopeMin_
                               << " top Level " << levelT1_ << ":" << levelT2_ << " FirstLayer " << firstLayer_;
#endif

    numberingScheme_ = std::make_unique<HGCalNumberingScheme>(*hgcons_, mydet_, nameX_, fileName_);
  } else {
    throw cms::Exception("Unknown", "HGCScintSD") << "Cannot find HGCalDDDConstants for " << nameX_ << "\n";
  }
}

void HGCScintSD::initRun() {}

bool HGCScintSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit_));
}

uint32_t HGCScintSD::setDetUnitId(int layer, int module, int cell, int iz, G4ThreeVector& pos) {
  uint32_t id = numberingScheme_ ? numberingScheme_->getUnitID(layer, module, cell, iz, pos, weight_) : 0;
  return id;
}

bool HGCScintSD::isItinFidVolume(const G4ThreeVector& pos) {
  if (fiducialCut_) {
    return (hgcons_->distFromEdgeTrap(pos.x(), pos.y(), pos.z()) > distanceFromEdge_);
  } else {
    return true;
  }
}
