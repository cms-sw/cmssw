///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.cc
// Description: Sensitive Detector class for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/FastMath.h"

#include "SimG4CMS/Calo/interface/HGCSD.h"
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

HGCSD::HGCSD(const std::string& name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme_(nullptr), mouseBite_(nullptr), slopeMin_(0), levelT_(99) {

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit_         = m_HGC.getParameter<double>("EminHit")*CLHEP::MeV;
  storeAllG4Hits_  = m_HGC.getParameter<bool>("StoreAllG4Hits");
  rejectMB_        = m_HGC.getParameter<bool>("RejectMouseBite");
  waferRot_        = m_HGC.getParameter<bool>("RotatedWafer");
  angles_          = m_HGC.getUntrackedParameter<std::vector<double>>("WaferAngles");
  double waferSize = m_HGC.getUntrackedParameter<double>("WaferSize")*CLHEP::mm;
  double mouseBite = m_HGC.getUntrackedParameter<double>("MouseBite")*CLHEP::mm;
  mouseBiteCut_    = waferSize*tan(30.0*CLHEP::deg) - mouseBite;

  if(storeAllG4Hits_) {
    setUseMap(false);
    setNumberCheckedHits(0);
  }
  //this is defined in the hgcsens.xml
  G4String myName = name;
  myFwdSubdet_= ForwardSubdetector::ForwardEmpty;
  nameX_ = "HGCal";
  if (myName.find("HitsEE")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCEE;
    nameX_        = "HGCalEESensitive";
  } else if (myName.find("HitsHEfront")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEF;
    nameX_       = "HGCalHESiliconSensitive";
  } else if (myName.find("HitsHEback")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEB;
    nameX_       = "HGCalHEScintillatorSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim")<< "**************************************************"
			    << "\n"
			    << "*                                                *"
			    << "\n"
			    << "* Constructing a HGCSD  with name " << name << "\n"
			    << "*                                                *"
			    << "\n"
			    << "**************************************************";
#endif
  edm::LogVerbatim("HGCSim") << "HGCSD:: Threshold for storing hits: " 
			     << eminHit << " for " << nameX_ << " subdet "
			     << myFwdSubdet_;
  edm::LogVerbatim("HGCSim") << "Flag for storing individual Geant4 Hits "
			     << storeAllG4Hits_;
  edm::LogVerbatim("HGCSim") << "Reject MosueBite Flag: " << rejectMB_ 
			     << " Size of wafer " << waferSize 
			     << " Mouse Bite " << mouseBite << ":"
			     << mouseBiteCut_ << " along " << angles_.size() 
			     << " axes";
}

HGCSD::~HGCSD() { 
  delete numberingScheme_;
  delete mouseBite_;
}

double HGCSD::getEnergyDeposit(const G4Step* aStep) {

  double r = aStep->GetPreStepPoint()->GetPosition().perp();
  double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());

#ifdef EDM_ML_DEBUG
  G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
  G4LogicalVolume* lv =
    aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
  edm::LogVerbatim("HGCSim") << "HGCSD: Hit from standard path from "
			     << lv->GetName() << " for Track " 
			     << aStep->GetTrack()->GetTrackID() << " ("
			     << aStep->GetTrack()->GetDefinition()->GetParticleName() 
			     << ") R = " << r << " Z = "
			     << z << " slope = " << r/z << ":" << slopeMin_;
#endif

  // Apply fiductial volume
  if (r < z*slopeMin_) { return 0.0; }
 
  double wt1    = getResponseWt(aStep->GetTrack());
  double wt2    = aStep->GetTrack()->GetWeight();
  double destep = wt1*aStep->GetTotalEnergyDeposit();
  if (wt2 > 0) destep *= wt2;
  return destep;
}

uint32_t HGCSD::setDetUnitId(const G4Step * aStep) { 

  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  //determine the exact position in global coordinates in the mass geometry 
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();
  float globalZ=touch->GetTranslation(0).z();
  int iz( globalZ>0 ? 1 : -1);

  //convert to local coordinates (=local to the current volume): 
  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  
  //get the det unit id with 
  ForwardSubdetector subdet = myFwdSubdet_;

  int layer, module, cell;
  if (touch->GetHistoryDepth() == levelT_) {
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
    layer  = touch->GetReplicaNumber(2);
    module = touch->GetReplicaNumber(1);
    cell   = touch->GetReplicaNumber(0);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "Depths: " << touch->GetHistoryDepth() 
			     << " name " << touch->GetVolume(0)->GetName() 
			     << ":" << touch->GetReplicaNumber(0) << "   "
			     << touch->GetVolume(1)->GetName() 
			     << ":" << touch->GetReplicaNumber(1) << "   "
			     << touch->GetVolume(2)->GetName() 
			     << ":" << touch->GetReplicaNumber(2) << "   "
			     << " layer:module:cell " << layer << ":"
			     << module << ":" << cell <<" Material "
			     << mat->GetName() << ":"
			     << aStep->GetPreStepPoint()->GetMaterial()->GetRadlen();
#endif
  // The following statement should be examined later before elimination
  // VI: this is likely a check if media is vacuum - not needed 
  if (aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() > 100000.) return 0;
  
  uint32_t id = setDetUnitId (subdet, layer, module, cell, iz, localpos);
  if (rejectMB_ && id != 0) {
    int det, z, lay, wafer, type, ic;
    HGCalTestNumbering::unpackHexagonIndex(id, det, z, lay, wafer, type, ic);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "ID " << std::hex << id << std::dec 
			       << " Decode " << det << ":" << z << ":" << lay
			       << ":" << wafer << ":" << type << ":" << ic;
#endif
    if (mouseBite_->exclude(hitPoint, z, wafer, 0)) id = 0;
  }
  return id;
}

void HGCSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<HGCalDDDConstants>    hdc;
  es->get<IdealGeometryRecord>().get(nameX_,hdc);
  if (hdc.isValid()) {
    const HGCalDDDConstants* hgcons = hdc.product();
    geom_mode_                = hgcons->geomMode();
    slopeMin_                 = hgcons->minSlope();
    levelT_                   = hgcons->levelTop();
    numberingScheme_          = new HGCNumberingScheme(*hgcons,nameX_);
    if (rejectMB_) mouseBite_ = new HGCMouseBite(*hgcons,angles_,mouseBiteCut_,waferRot_);
  } else {
    edm::LogError("HGCSim") << "HCalSD : Cannot find HGCalDDDConstants for "
			    << nameX_;
    throw cms::Exception("Unknown", "HGCSD") 
      << "Cannot find HGCalDDDConstants for " << nameX_ << "\n";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCSD::Initialized with mode " << geom_mode_ 
			     << " Slope cut " << slopeMin_ << " top Level "
			     << levelT_;
#endif
}

void HGCSD::initRun() {
}

bool HGCSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit_));
}

uint32_t HGCSD::setDetUnitId (ForwardSubdetector &subdet, int layer, int module,
			      int cell, int iz, G4ThreeVector &pos) {  
  uint32_t id = numberingScheme_ ? 
    numberingScheme_->getUnitID(subdet, layer, module, cell, iz, pos) : 0;
  return id;
}

