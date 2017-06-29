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

HGCSD::HGCSD(G4String name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme(0), mouseBite_(0), slopeMin_(0), levelT_(99) {

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit          = m_HGC.getParameter<double>("EminHit")*CLHEP::MeV;
  storeAllG4Hits_  = m_HGC.getParameter<bool>("StoreAllG4Hits");
  rejectMB_        = m_HGC.getParameter<bool>("RejectMouseBite");
  waferRot_        = m_HGC.getParameter<bool>("RotatedWafer");
  angles_          = m_HGC.getUntrackedParameter<std::vector<double>>("WaferAngles");
  double waferSize = m_HGC.getUntrackedParameter<double>("WaferSize")*CLHEP::mm;
  double mouseBite = m_HGC.getUntrackedParameter<double>("MouseBite")*CLHEP::mm;
  mouseBiteCut_    = waferSize*tan(30.0*CLHEP::deg) - mouseBite;

  //this is defined in the hgcsens.xml
  G4String myName(this->nameOfSD());
  myFwdSubdet_= ForwardSubdetector::ForwardEmpty;
  nameX = "HGCal";
  if (myName.find("HitsEE")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCEE;
    nameX        = "HGCalEESensitive";
  } else if (myName.find("HitsHEfront")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEF;
    nameX        = "HGCalHESiliconSensitive";
  } else if (myName.find("HitsHEback")!=std::string::npos) {
    myFwdSubdet_ = ForwardSubdetector::HGCHEB;
    nameX        = "HGCalHEScintillatorSensitive";
  }

#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCSim")<< "**************************************************"
			<< "\n"
			<< "*                                                *"
			<< "\n"
			<< "* Constructing a HGCSD  with name " << name << "\n"
			<< "*                                                *"
			<< "\n"
			<< "**************************************************";
#endif
  edm::LogInfo("HGCSim") << "HGCSD:: Threshold for storing hits: " << eminHit
			 << " for " << nameX << " subdet " << myFwdSubdet_;
  edm::LogInfo("HGCSim") << "Flag for storing individual Geant4 Hits "
			 << storeAllG4Hits_;
  edm::LogInfo("HGCSim") << "Reject MosueBite Flag: " << rejectMB_ 
			 << " Size of wafer " << waferSize << " Mouse Bite "
			 << mouseBite << ":" << mouseBiteCut_ << " along "
			 << angles_.size() << " axes";
}

HGCSD::~HGCSD() { 
  if (numberingScheme)  delete numberingScheme;
  if (mouseBite_)       delete mouseBite_;
}

bool HGCSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    double r = aStep->GetPreStepPoint()->GetPosition().perp();
    double z = std::abs(aStep->GetPreStepPoint()->GetPosition().z());
#ifdef EDM_ML_DEBUG
    G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
    bool notaMuon = (parCode == mupPDG || parCode == mumPDG ) ? false : true;
    G4LogicalVolume* lv =
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
    edm::LogInfo("HGCSim") << "HGCSD: Hit from standard path from "
			   << lv->GetName() << " for Track " 
			   << aStep->GetTrack()->GetTrackID() << " ("
			   << aStep->GetTrack()->GetDefinition()->GetParticleName() 
			   << ":" << notaMuon << ") R = " << r << " Z = " << z
			   << " slope = " << r/z << ":" << slopeMin_;
#endif
    // Apply fiducial cuts
    if (r/z >= slopeMin_) {
      if (getStepInfo(aStep)) {
	if ((storeAllG4Hits_ || (hitExists() == false)) && 
	    (edepositEM+edepositHAD>0.)) currentHit = createNewHit();
      }
    }
    return true;
  }
} 

double HGCSD::getEnergyDeposit(G4Step* aStep) {
  double wt1    = getResponseWt(aStep->GetTrack());
  double wt2    = aStep->GetTrack()->GetWeight();
  double destep = wt1*(aStep->GetTotalEnergyDeposit());
  if (wt2 > 0) destep *= wt2;
  return destep;
}

uint32_t HGCSD::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  //determine the exact position in global coordinates in the mass geometry 
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();
  float globalZ=touch->GetTranslation(0).z();
  int iz( globalZ>0 ? 1 : -1);

  //convert to local coordinates (=local to the current volume): 
  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  
  //get the det unit id with 
  ForwardSubdetector subdet = myFwdSubdet_;

  int layer(0), module(0), cell(0);
  if (m_mode == HGCalGeometryMode::Square) {
    layer  = touch->GetReplicaNumber(0);
    module = touch->GetReplicaNumber(1);
  } else {
    if (touch->GetHistoryDepth() == levelT_) {
      layer  = touch->GetReplicaNumber(0);
      module = -1;
      cell   = -1;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("HGCSim") << "Depths: " << touch->GetHistoryDepth() 
			     << " name " << touch->GetVolume(0)->GetName() 
			     << " layer:module:cell " << layer << ":" 
			     << module << ":" << cell << std::endl;
#endif
    } else {
      layer  = touch->GetReplicaNumber(2);
      module = touch->GetReplicaNumber(1);
      cell   = touch->GetReplicaNumber(0);
    }
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HGCSim") << "Depths: " << touch->GetHistoryDepth() <<" name "
			   << touch->GetVolume(0)->GetName() 
			   << ":" << touch->GetReplicaNumber(0) << "   "
			   << touch->GetVolume(1)->GetName() 
			   << ":" << touch->GetReplicaNumber(1) << "   "
			   << touch->GetVolume(2)->GetName() 
			   << ":" << touch->GetReplicaNumber(2) << "   "
			   << " layer:module:cell " << layer << ":" << module 
			   << ":" << cell <<" Material " << mat->GetName()<<":"
			   << aStep->GetPreStepPoint()->GetMaterial()->GetRadlen()
			   << std::endl;
#endif
    if (aStep->GetPreStepPoint()->GetMaterial()->GetRadlen() > 100000.) return 0;
  }

  uint32_t id = setDetUnitId (subdet, layer, module, cell, iz, localpos);
  if (rejectMB_ && m_mode != HGCalGeometryMode::Square && id != 0) {
    int det, z, lay, wafer, type, ic;
    HGCalTestNumbering::unpackHexagonIndex(id, det, z, lay, wafer, type, ic);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HGCSim") << "ID " << std::hex << id << std::dec << " Decode "
			   << det << ":" << z << ":" << lay << ":" << wafer 
			   << ":" << type << ":" << ic << std::endl;
#endif
    if (mouseBite_->exclude(hitPoint, z, wafer)) id = 0;
  }
  return id;
}

void HGCSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<HGCalDDDConstants>    hdc;
  es->get<IdealGeometryRecord>().get(nameX,hdc);
  if (hdc.isValid()) {
    const HGCalDDDConstants* hgcons = hdc.product();
    m_mode                    = hgcons->geomMode();
    slopeMin_                 = hgcons->minSlope();
    levelT_                   = hgcons->levelTop();
    numberingScheme           = new HGCNumberingScheme(*hgcons,nameX);
    if (rejectMB_) mouseBite_ = new HGCMouseBite(*hgcons,angles_,mouseBiteCut_,waferRot_);
  } else {
    edm::LogError("HGCSim") << "HCalSD : Cannot find HGCalDDDConstants for "
			    << nameX;
    throw cms::Exception("Unknown", "HGCSD") << "Cannot find HGCalDDDConstants for " << nameX << "\n";
  }
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCSim") << "HGCSD::Initialized with mode " << m_mode 
			 << " Slope cut " << slopeMin_ << " top Level "
			 << levelT_ << std::endl;
#endif
}

void HGCSD::initRun() {
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String          particleName;
  mumPDG = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HGCSim") << "HGCSD: Particle code for mu- = " << mumPDG
			 << " for mu+ = " << mupPDG;
#endif
}

bool HGCSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit));
}

uint32_t HGCSD::setDetUnitId (ForwardSubdetector &subdet, int layer, int module,
			      int cell, int iz, G4ThreeVector &pos) {  
  uint32_t id = numberingScheme ? 
    numberingScheme->getUnitID(subdet, layer, module, cell, iz, pos) : 0;
  return id;
}

int HGCSD::setTrackID (G4Step* aStep) {
  theTrack     = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HGCSim") << "HGCSD: Problem with primaryID **** set by "
			   << "force to TkID **** " <<theTrack->GetTrackID();
#endif
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID())
    resetForNewPrimary(preStepPoint->GetPosition(), etrack);

  return primaryID;
}
