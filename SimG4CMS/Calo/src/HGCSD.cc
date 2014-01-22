///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.cc
// Description: Sensitive Detector class for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

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

#include <iostream>
#include <fstream>
#include <iomanip>

#define DebugLog

HGCSD::HGCSD(G4String name, const DDCompactView & cpv,
	     SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<int>("TimeSliceUnit"),
         p.getParameter<edm::ParameterSet>("HGCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme(0) {

  edm::ParameterSet m_HGC = p.getParameter<edm::ParameterSet>("HGCSD");
  eminHit          = m_HGC.getParameter<double>("EminHit")*MeV;

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

  std::string attribute, value;
  // Constants for Numbering Scheme
  attribute = "Volume";
  value     = "HGC";
  DDSpecificsFilter filter0;
  DDValue           ddv0(attribute, value, 0);
  filter0.setCriteria(ddv0, DDSpecificsFilter::equals);
  DDFilteredView fv0(cpv);
  fv0.addFilter(filter0);
  fv0.firstChild();
  DDsvalues_type sv0(fv0.mergedSpecifics());

  gpar    = getDDDArray("GeomParHGC",sv0);
  numberingScheme = new HGCNumberingScheme(gpar);
}

HGCSD::~HGCSD() { 
  if (numberingScheme)  delete numberingScheme;
}

bool HGCSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
    bool notaMuon = (parCode == mupPDG || parCode == mumPDG ) ? false : true;
#ifdef DebugLog
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
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();

  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  int iz     = (hitPoint.z() > 0) ? 1 : -1;
  int subdet = (touch->GetReplicaNumber(4));
  int module = (touch->GetReplicaNumber(3));
  int layer  = (touch->GetReplicaNumber(2));

  return setDetUnitId (subdet, localpos, iz, module, layer);
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


uint32_t HGCSD::setDetUnitId (int subdet, G4ThreeVector pos, int iz, int mod, 
			      int layer) {
  uint32_t id = 0;
  //get the ID
  if (numberingScheme) id = numberingScheme->getUnitID(subdet, pos, iz, mod, 
						       layer);
  return id;
}

std::vector<double> HGCSD::getDDDArray(const std::string & str,
                                        const DDsvalues_type & sv) {
#ifdef DebugLog
  LogDebug("HGCSim") << "HGCSD:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HGCSim") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("HGCSim") << "HGCSD : # of " << str << " bins " << nval
			      << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HGCSD") << "nval < 2 for array " << str << "\n";
    }
    
    return fvec;
  } else {
    edm::LogError("HGCSim") << "HGCSD :  cannot get array " << str;
    throw cms::Exception("Unknown", "HGCSD") << "cannot get array " << str << "\n";
  }
}

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
