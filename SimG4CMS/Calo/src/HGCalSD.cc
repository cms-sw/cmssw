///////////////////////////////////////////////////////////////////////////////
// File: HGCalSD.cc
// Description: Sensitive Detector class for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/Math/interface/FastMath.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "SimG4CMS/Calo/interface/HGCalSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
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

using namespace angle_units::operators;

HGCalSD::HGCalSD(const std::string& name,
                 const HGCalDDDConstants* hgc,
                 const SensitiveDetectorCatalog& clg,
                 edm::ParameterSet const& p,
                 const SimTrackManager* manager)
    : CaloSD(name,
             clg,
             p,
             manager,
             static_cast<float>(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")),
      hgcons_(hgc),
      slopeMin_(0),
      levelT1_(99),
      levelT2_(99),
      useSimWt_(0),
      tan30deg_(std::tan(30.0 * CLHEP::deg)),
      cos30deg_(std::cos(30.0 * CLHEP::deg)) {
  numberingScheme_.reset(nullptr);
  guardRing_.reset(nullptr);
  mouseBite_.reset(nullptr);

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit_ = m_HGC.getParameter<double>("EminHit") * CLHEP::MeV;
  fiducialCut_ = m_HGC.getParameter<bool>("FiducialCut");
  storeAllG4Hits_ = m_HGC.getParameter<bool>("StoreAllG4Hits");
  rejectMB_ = m_HGC.getParameter<bool>("RejectMouseBite");
  waferRot_ = m_HGC.getParameter<bool>("RotatedWafer");
  cornerMinMask_ = m_HGC.getParameter<int>("CornerMinMask");
  angles_ = m_HGC.getUntrackedParameter<std::vector<double>>("WaferAngles");
  missingFile_ = m_HGC.getUntrackedParameter<std::string>("MissingWaferFile");
  checkID_ = m_HGC.getUntrackedParameter<bool>("CheckID");
  verbose_ = m_HGC.getUntrackedParameter<int>("Verbosity");

  if (storeAllG4Hits_) {
    setUseMap(false);
    setNumberCheckedHits(0);
  }

  //this is defined in the hgcsens.xml
  G4String myName = name;
  mydet_ = DetId::Forward;
  nameX_ = "HGCal";
  if (myName.find("HitsEE") != std::string::npos) {
    mydet_ = DetId::HGCalEE;
    nameX_ = "HGCalEESensitive";
  } else if (myName.find("HitsHEfront") != std::string::npos) {
    mydet_ = DetId::HGCalHSi;
    nameX_ = "HGCalHESiliconSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "**************************************************"
                             << "\n"
                             << "*                                                *"
                             << "\n"
                             << "* Constructing a HGCalSD  with name " << name << "\n"
                             << "*                                                *"
                             << "\n"
                             << "**************************************************";
#endif
  edm::LogVerbatim("HGCSim") << "HGCalSD:: Threshold for storing hits: " << eminHit_ << " for " << nameX_
                             << " detector " << mydet_;
  edm::LogVerbatim("HGCSim") << "Flag for storing individual Geant4 Hits " << storeAllG4Hits_;
  edm::LogVerbatim("HGCSim") << "Fiducial volume cut with cut from eta/phi "
                             << "boundary " << fiducialCut_ << " at " << distanceFromEdge_;
  edm::LogVerbatim("HGCSim") << "Reject MosueBite Flag: " << rejectMB_ << " cuts along " << angles_.size()
                             << " axes: " << angles_[0] << ", " << angles_[1];
}

double HGCalSD::getEnergyDeposit(const G4Step* aStep) {
  double r = aStep->GetPreStepPoint()->GetPosition().perp();
  double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());
#ifdef EDM_ML_DEBUG
  G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4String parName = aStep->GetTrack()->GetDefinition()->GetParticleName();
  G4LogicalVolume* lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  edm::LogVerbatim("HGCSim") << "HGCalSD: Hit from standard path from " << lv->GetName() << " for Track "
                             << aStep->GetTrack()->GetTrackID() << " (" << parCode << ":" << parName << ") R = " << r
                             << " Z = " << z << " slope = " << r / z << ":" << slopeMin_;
#endif
  // Apply fiducial cut
  if (r < z * slopeMin_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCalSD: Fiducial Volume cut";
#endif
    return 0.0;
  }

  double wt1 = getResponseWt(aStep->GetTrack());
  double wt2 = aStep->GetTrack()->GetWeight();
  double destep = weight_ * wt1 * (aStep->GetTotalEnergyDeposit());
  if (wt2 > 0)
    destep *= wt2;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalSD: weights= " << weight_ << ":" << wt1 << ":" << wt2 << " Total weight "
                             << weight_ * wt1 * wt2 << " deStep: " << aStep->GetTotalEnergyDeposit() << ":" << destep;
#endif
  return destep;
}

uint32_t HGCalSD::setDetUnitId(const G4Step* aStep) {
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

  int layer(0), moduleLev(-1), cell(-1);
  if (useSimWt_ > 0) {
    layer = touch->GetReplicaNumber(2);
    moduleLev = 1;
  } else if (touch->GetHistoryDepth() > levelT2_) {
    layer = touch->GetReplicaNumber(4);
    cell = touch->GetReplicaNumber(1);
    moduleLev = 3;
  } else {
    layer = touch->GetReplicaNumber(3);
    moduleLev = 2;
  }
  int module = touch->GetReplicaNumber(moduleLev);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "DepthsTop: " << touch->GetHistoryDepth() << ":" << levelT1_ << ":" << levelT2_ << ":"
                             << useSimWt_ << " name " << touch->GetVolume(0)->GetName() << " layer:module:cell "
                             << layer << ":" << moduleLev << ":" << module << ":" << cell;
  printDetectorLevels(touch);
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
  if ((rejectMB_ || fiducialCut_) && id != 0) {
    auto uv = HGCSiliconDetId(id).waferUV();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "ID " << std::hex << id << std::dec << " " << HGCSiliconDetId(id);
#endif
    G4ThreeVector local = (touch->GetHistory()->GetTransform(moduleLev).TransformPoint(hitPoint));
    if (fiducialCut_) {
      int layertype = hgcons_->layerType(layer);
      int frontBack = HGCalTypes::layerFrontBack(layertype);
      if (guardRing_->exclude(local, iz, frontBack, layer, uv.first, uv.second)) {
        id = 0;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "Rejected by GuardRing cutoff *****";
#endif
      }
    }
    if ((rejectMB_) && (mouseBite_->exclude(local, iz, layer, uv.first, uv.second))) {
      id = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "Rejected by MouseBite cutoff *****";
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  if (id != 0)
    edm::LogVerbatim("HGCSim") << HGCSiliconDetId(id);
#endif
  if ((id != 0) && checkID_) {
    HGCSiliconDetId hid1(id);
    bool cshift = (hgcons_->cassetteShiftSilicon(hid1.zside(), hid1.layer(), hid1.waferU(), hid1.waferV()));
    std::string_view pid = (cshift ? "HGCSim" : "HGCalSim");
    bool debug = (verbose_ > 0) ? true : false;
    auto xy = hgcons_->locateCell(hid1, debug);
    double xx = (hid1.zside() > 0) ? xy.first : -xy.first;
    double dx = xx - (hitPoint.x() / CLHEP::cm);
    double dy = xy.second - (hitPoint.y() / CLHEP::cm);
    double diff = (dx * dx + dy * dy);
    constexpr double tol = 2.0 * 2.0;
    bool valid1 = hgcons_->isValidHex8(hid1.layer(), hid1.waferU(), hid1.waferV(), hid1.cellU(), hid1.cellV(), true);
    if ((diff > tol) || (!valid1))
      pid = "HGCalError";
    auto partn = hgcons_->waferTypeRotation(hid1.layer(), hid1.waferU(), hid1.waferV(), false, false);
    int indx = HGCalWaferIndex::waferIndex(layer, hid1.waferU(), hid1.waferV());
    edm::LogVerbatim(pid) << "CheckID " << HGCSiliconDetId(id) << " Layer:Module:Cell:ModuleLev " << layer << ":"
                          << module << ":" << cell << ":" << moduleLev << " SimWt:history " << useSimWt_ << ":"
                          << touch->GetHistoryDepth() << ":" << levelT1_ << ":" << levelT2_ << " input position: ("
                          << hitPoint.x() / CLHEP::cm << ", " << hitPoint.y() / CLHEP::cm << ":"
                          << convertRadToDeg(std::atan2(hitPoint.y(), hitPoint.x())) << "); position from ID (" << xx
                          << ", " << xy.second << ") distance " << dx << ":" << dy << ":" << std::sqrt(diff)
                          << " Valid " << valid1 << " Wafer type|rotation " << partn.first << ":" << partn.second
                          << " Part:Orient:Cassette " << std::get<1>(hgcons_->waferFileInfo(indx)) << ":"
                          << std::get<2>(hgcons_->waferFileInfo(indx)) << ":"
                          << std::get<3>(hgcons_->waferFileInfo(indx)) << " CassetteShift " << cshift;
    if ((diff > tol) || (!valid1)) {
      printDetectorLevels(touch);
      hgcons_->locateCell(hid1, true);
    }
  }
  return id;
}

void HGCalSD::update(const BeginOfJob* job) {
  if (hgcons_ != nullptr) {
    geom_mode_ = hgcons_->geomMode();
    slopeMin_ = hgcons_->minSlope();
    levelT1_ = hgcons_->levelTop(0);
    levelT2_ = hgcons_->levelTop(1);
    useSimWt_ = hgcons_->getParameter()->useSimWt_;
    int useOffset = hgcons_->getParameter()->useOffset_;
    double waferSize = hgcons_->waferSize(false);
    double mouseBite = hgcons_->mouseBite(false);
    double guardRingOffset = hgcons_->guardRingOffset(false);
    double sensorSizeOffset = hgcons_->sensorSizeOffset(false);
    if (useOffset > 0) {
      rejectMB_ = true;
      fiducialCut_ = true;
    }
    double mouseBiteNew = (fiducialCut_) ? (mouseBite + guardRingOffset + sensorSizeOffset / cos30deg_) : mouseBite;
    mouseBiteCut_ = waferSize * tan30deg_ - mouseBiteNew;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCalSD::Initialized with mode " << geom_mode_ << " Slope cut " << slopeMin_
                               << " top Level " << levelT1_ << ":" << levelT2_ << " useSimWt " << useSimWt_ << " wafer "
                               << waferSize << ":" << mouseBite << ":" << guardRingOffset << ":" << sensorSizeOffset
                               << ":" << mouseBiteNew << ":" << mouseBiteCut_ << " useOffset " << useOffset;
#endif

    numberingScheme_ = std::make_unique<HGCalNumberingScheme>(*hgcons_, mydet_, nameX_, missingFile_);
    if (rejectMB_)
      mouseBite_ = std::make_unique<HGCMouseBite>(*hgcons_, angles_, mouseBiteCut_, waferRot_);
    if (fiducialCut_)
      guardRing_ = std::make_unique<HGCGuardRing>(*hgcons_);
  } else {
    throw cms::Exception("Unknown", "HGCalSD") << "Cannot find HGCalDDDConstants for " << nameX_ << "\n";
  }
}

void HGCalSD::initRun() {}

bool HGCalSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit_));
}

uint32_t HGCalSD::setDetUnitId(int layer, int module, int cell, int iz, G4ThreeVector& pos) {
  uint32_t id = numberingScheme_ ? numberingScheme_->getUnitID(layer, module, cell, iz, pos, weight_) : 0;
  if (cornerMinMask_ > 2) {
    if (hgcons_->maskCell(DetId(id), cornerMinMask_)) {
      id = 0;
      ignoreRejection();
    }
  }
  if (hgcons_->waferHexagon8File() || (id == 0))
    ignoreRejection();
  return id;
}
