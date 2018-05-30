///////////////////////////////////////////////////////////////////////////////
// File: HGCalSD.cc
// Description: Sensitive Detector class for High Granularity Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/FastMath.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "SimG4CMS/Calo/interface/HGCalSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"
#include "G4Trap.hh"

#include <iostream>
#include <fstream>
#include <iomanip>

//#define EDM_ML_DEBUG

HGCalSD::HGCalSD(const std::string& name, const DDCompactView & cpv,
		 const SensitiveDetectorCatalog & clg, 
		 edm::ParameterSet const & p, const SimTrackManager* manager):
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme_(nullptr), mouseBite_(nullptr), slopeMin_(0), levelT1_(99), 
  levelT2_(99), isScint_(false), tan30deg_(std::tan(30.0*CLHEP::deg)) {

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit_         = m_HGC.getParameter<double>("EminHit")*CLHEP::MeV;
  useBirk_         = m_HGC.getParameter<bool>("UseBirkLaw");
  birk1_           = m_HGC.getParameter<double>("BirkC1")*(CLHEP::g/(CLHEP::MeV*CLHEP::cm2));
  birk2_           = m_HGC.getParameter<double>("BirkC2");
  birk3_           = m_HGC.getParameter<double>("BirkC3");
  storeAllG4Hits_  = m_HGC.getParameter<bool>("StoreAllG4Hits");
  rejectMB_        = m_HGC.getParameter<bool>("RejectMouseBite");
  waferRot_        = m_HGC.getParameter<bool>("RotatedWafer");
  angles_          = m_HGC.getUntrackedParameter<std::vector<double>>("WaferAngles");

  if(storeAllG4Hits_) {
    setUseMap(false);
    setNumberCheckedHits(0);
  }

  //this is defined in the hgcsens.xml
  G4String myName = name;
  mydet_ = DetId::Forward;
  nameX_ = "HGCal";
  if (myName.find("HitsEE")!=std::string::npos) {
    mydet_  = DetId::HGCalEE;
    nameX_  = "HGCalEESensitive";
  } else if (myName.find("HitsHEfront")!=std::string::npos) {
    mydet_  = DetId::HGCalHSi;
    nameX_  = "HGCalHESiliconSensitive";
  } else if (myName.find("HitsHEback")!=std::string::npos) {
    mydet_  = DetId::HGCalHSc;
    nameX_  = "HGCalHEScintillatorSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim")<< "**************************************************"
			    << "\n"
			    << "*                                                *"
			    << "\n"
			    << "* Constructing a HGCalSD  with name " << name << "\n"
			    << "*                                                *"
			    << "\n"
			    << "**************************************************";
#endif
  edm::LogVerbatim("HGCSim") << "HGCalSD:: Threshold for storing hits: " 
			     << eminHit_ << " for " << nameX_ << " detector "
			     << mydet_;
  edm::LogVerbatim("HGCSim") << "Flag for storing individual Geant4 Hits "
			     << storeAllG4Hits_;
  edm::LogVerbatim("HGCSim") << "Reject MosueBite Flag: " << rejectMB_ 
			     << " cuts along " << angles_.size() << " axes: "
			     << angles_[0] << ", " << angles_[1];
}

HGCalSD::~HGCalSD() { 
  delete numberingScheme_;
  delete mouseBite_;
}

double HGCalSD::getEnergyDeposit(const G4Step* aStep) {

  double r = aStep->GetPreStepPoint()->GetPosition().perp();
  double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());
#ifdef EDM_ML_DEBUG
  G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4LogicalVolume* lv =
    aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  edm::LogVerbatim("HGCSim") << "HGCalSD: Hit from standard path from "
			     << lv->GetName() << " for Track " 
			     << aStep->GetTrack()->GetTrackID() << " ("
			     << aStep->GetTrack()->GetDefinition()->GetParticleName() 
			     << ") R = " << r << " Z = "
			     << z << " slope = " << r/z << ":" << slopeMin_;
#endif
  // Apply fiducial cut
  if (r < z*slopeMin_) { return 0.0; }

  double wt1    = getResponseWt(aStep->GetTrack());
  double wt2    = aStep->GetTrack()->GetWeight();
  double wt3    = ((useBirk_ && isScint_) ? 
		   getAttenuation(aStep, birk1_, birk2_, birk3_) : 1.0);
  double destep = weight_*wt1*wt3*(aStep->GetTotalEnergyDeposit());
  if (wt2 > 0) destep *= wt2;
#ifdef EDM_ML_DEBUG
  edm::LogWarning("HGCalSim")  << "HGCalSD: weights= " << weight_ << ":" << wt1 << ":"
			       << wt2 << ":" << wt3 << " Total weight "
			       << weight_*wt1*wt2*wt3 << " deStep: "
			       << aStep->GetTotalEnergyDeposit() << ":" <<destep;
#endif
  return destep;
}

uint32_t HGCalSD::setDetUnitId(const G4Step * aStep) { 

  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  //determine the exact position in global coordinates in the mass geometry 
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();
  float globalZ = touch->GetTranslation(0).z();
  int iz( globalZ>0 ? 1 : -1);

  int layer, module, cell;
  if ((touch->GetHistoryDepth() == levelT1_) || 
      (touch->GetHistoryDepth() == levelT2_)) {
    layer  = touch->GetReplicaNumber(0);
    module = -1;
    cell   = -1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth() 
			       << " name " << touch->GetVolume(0)->GetName() 
			       << " layer:module:cell " << layer << ":" 
			       << module << ":" << cell;
#endif
  } else {
    layer  = touch->GetReplicaNumber(3);
    module = touch->GetReplicaNumber(2);
    cell   = touch->GetReplicaNumber(1);
  }
#ifdef EDM_ML_DEBUG
  G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
  edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth()
			     << " name " << touch->GetVolume(0)->GetName() 
			     << ":" << touch->GetReplicaNumber(0) << "   "
			     << touch->GetVolume(1)->GetName() 
			     << ":" << touch->GetReplicaNumber(1) << "   "
			     << touch->GetVolume(2)->GetName() 
			     << ":" << touch->GetReplicaNumber(2) << "   "
			     << touch->GetVolume(3)->GetName() 
			     << ":" << touch->GetReplicaNumber(3) << "   "
			     << touch->GetVolume(4)->GetName() 
			     << ":" << touch->GetReplicaNumber(4) << "   "
			     << " layer:module:cell " << layer << ":" 
			     << module  << ":" << cell << " Material " 
			     << mat->GetName() << ":" << mat->GetRadlen();
#endif
  // The following statement should be examined later before elimination
  if (aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() > 100000.) return 0;
  
  uint32_t id = setDetUnitId (layer, module, cell, iz, hitPoint);
  if (rejectMB_ && id != 0 && (!isScint_)) {
    auto uv = HGCSiliconDetId(id).waferUV();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "ID " << std::hex << id << std::dec 
			       << " " << HGCSiliconDetId(id);
#endif
    if (mouseBite_->exclude(hitPoint, iz, uv.first, uv.second)) {
      id = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "Rejected by mousebite cutoff *****";
#endif
    }
  }
  return id;
}

void HGCalSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<HGCalDDDConstants>    hdc;
  es->get<IdealGeometryRecord>().get(nameX_,hdc);
  if (hdc.isValid()) {
    const HGCalDDDConstants* hgcons = hdc.product();
    geom_mode_       = hgcons->geomMode();
    slopeMin_        = hgcons->minSlope();
    levelT1_         = hgcons->levelTop(0);
    levelT2_         = hgcons->levelTop(1);
    isScint_         = (geom_mode_ == HGCalGeometryMode::Trapezoid);
    double waferSize = hgcons->waferSize(false);
    double mouseBite = hgcons->mouseBite(false);
    mouseBiteCut_    = waferSize*tan30deg_ - mouseBite;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCalSD::Initialized with mode " << geom_mode_ 
			       << " Slope cut " << slopeMin_ << " top Level "
			       << levelT1_ << ":" << levelT2_ << " isScint "
			       << isScint_ << " wafer " << waferSize << ":"
			       << mouseBite;
#endif

    numberingScheme_ = new HGCalNumberingScheme(*hgcons,mydet_,nameX_);
    if (rejectMB_ && (!isScint_)) 
      mouseBite_ = new HGCMouseBite(*hgcons,angles_,mouseBiteCut_,waferRot_);
  } else {
    throw cms::Exception("Unknown", "HGCalSD") << "Cannot find HGCalDDDConstants for " << nameX_ << "\n";
  }
}

void HGCalSD::initRun() {
}

bool HGCalSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit_));
}

uint32_t HGCalSD::setDetUnitId (int layer, int module, int cell, int iz, 
				G4ThreeVector &pos) {  
  uint32_t id = numberingScheme_ ? 
    numberingScheme_->getUnitID(layer, module, cell, iz, pos, weight_) : 0;
  return id;
}

