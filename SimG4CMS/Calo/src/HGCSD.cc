///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.cc
// Description: Sensitive Detector class for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/FastMath.h"

#include "SimG4CMS/Calo/interface/HGCSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

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

//#define DebugLog

HGCSD::HGCSD(G4String name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("HGCSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme(0) {

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit          = m_HGC.getParameter<double>("EminHit")*CLHEP::MeV;
  bool checkID     = m_HGC.getUntrackedParameter<bool>("CheckID", false);
  verbosity        = m_HGC.getUntrackedParameter<int>("Verbosity",0);

  //this is defined in the hgcsens.xml
  G4String myName(this->nameOfSD());
  myFwdSubdet_= ForwardSubdetector::ForwardEmpty;
  std::string nameX("HGCal");
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

#ifdef DebugLog
  LogDebug("HGCSim") << "**************************************************" 
                      << "\n"
                      << "*                                                *"
                      << "\n"
                      << "* Constructing a HGCSD  with name " << name << "\n"
                      << "*                                                *"
                      << "\n"
                      << "**************************************************";
#endif
  edm::LogInfo("HGCSim") << "HGCSD:: Threshold for storing hits: " << eminHit;

  numberingScheme = new HGCNumberingScheme(cpv,nameX,checkID,verbosity);
}

HGCSD::~HGCSD() { 
  if (numberingScheme)  delete numberingScheme;
}

bool HGCSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
#ifdef DebugLog
    G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
    bool notaMuon = (parCode == mupPDG || parCode == mumPDG ) ? false : true;
    G4LogicalVolume* lv =
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
    edm::LogInfo("HGCSim") << "HGCSD: Hit from standard path from "
			   << lv->GetName() << " for Track " 
			   << aStep->GetTrack()->GetTrackID() << " ("
			   << aStep->GetTrack()->GetDefinition()->GetParticleName() 
			   << ":" << notaMuon << ")";
#endif
    if (getStepInfo(aStep)) {
      if (hitExists() == false && edepositEM+edepositHAD>0.) currentHit = createNewHit();
    }
    return true;
  }
} 

double HGCSD::getEnergyDeposit(G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
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

  int layer  = touch->GetReplicaNumber(0);
  int module = touch->GetReplicaNumber(1);
  if (verbosity > 0) 
    std::cout << "HGCSD::Global " << hitPoint << " local " << localpos 
	      << std::endl;
  return setDetUnitId (subdet, layer, module, iz, localpos);
}

void HGCSD::initRun() {
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String          particleName;
  mumPDG = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
#ifdef DebugLog
  LogDebug("HGCSim") << "HGCSD: Particle code for mu- = " << mumPDG
		     << " for mu+ = " << mupPDG;
#endif
}

bool HGCSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit));
}


//
uint32_t HGCSD::setDetUnitId (ForwardSubdetector &subdet, int &layer, int &module, int &iz, G4ThreeVector &pos) {  
  return (numberingScheme ? numberingScheme->getUnitID(subdet, layer, module, iz, pos) : 0);
}

//
int HGCSD::setTrackID (G4Step* aStep) {
  theTrack     = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
#ifdef DebugLog
    edm::LogInfo("HGCSim") << "HGCSD: Problem with primaryID **** set by "
			   << "force to TkID **** " <<theTrack->GetTrackID();
#endif
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID())
    resetForNewPrimary(preStepPoint->GetPosition(), etrack);

  return primaryID;
}
