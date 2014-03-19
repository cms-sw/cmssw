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

  //this is defined in the hgcsens.xml
  G4String myName(this->nameOfSD());
  myFwdSubdet_=ForwardSubdetector::ForwardEmpty;
  if(myName.find("HitsEE")!=std::string::npos) myFwdSubdet_=ForwardSubdetector::HGCEE;
  else if(myName.find("HitsHE")!=std::string::npos) myFwdSubdet_=ForwardSubdetector::HGCHE;

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
  std::vector<double>numberingPar(1,gpar[0]);
  if(myFwdSubdet_==ForwardSubdetector::HGCHE) numberingPar[0]=gpar[1];
  numberingScheme = new HGCNumberingScheme(numberingPar);
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

  //determine the exact position in global coordinates in the mass geometry 
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();

  //convert to local coordinates (=local to the current volume): 
  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

  //the solid of this detector
  G4VSolid *solid = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetSolid();
  G4Trap *layerSolid=(G4Trap *)solid;
    
  //FIXME urgently! no string parsing if possible
  //  G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();
  ForwardSubdetector fwdSubdet(ForwardSubdetector::HGCEE);
  //  if(nameVolume.find("HE")!=std::string::npos) fwdSubdet=ForwardSubdetector::HGCHE;
  //  size_t pos=nameVolume.find("_")+1;
  //  G4String layerStr=nameVolume.substr(pos,nameVolume.size()-1);
  //  G4int copyNb=preStepPoint->GetPhysicalVolume()->GetCopyNo();

  
  float dz(0), bl1(0),tl1(0),h1(0);
  if(layerSolid){
    dz =layerSolid->GetZHalfLength();   //half width of the layer
    bl1=layerSolid->GetXHalfLength1();  //half x length of the side at -h1
    tl1=layerSolid->GetXHalfLength2();  //half x length of the side at +h1
    h1=layerSolid->GetYHalfLength1();   //half height of the side
  }
  else{
    edm::LogError("HGCSim") << "[HGCSD] Failed to cast sensitive volume to trapezoid!! The DetIds will be missing lateral segmentation";
    //throw cms::Exception("Unknown", "HGCSD") <<  "[HGCSD] Failed to cast sensitive volume to trapezoid!! The DetIds will be missing lateral segmentation\n";
  }

  //get the det unit id with 
  ForwardSubdetector subdet =  fwdSubdet;
  //  int layer  = atoi(layerStr.c_str());
  //  int module = copyNb;
  //  int iz     = (hitPoint.z() > 0) ? 1 : -1;

  int layer  = touch->GetReplicaNumber(0);
  int module = touch->GetReplicaNumber(1);
  int iz     = touch->GetReplicaNumber(3)==1 ? 1 : -1;
  
  //  std::cout << "layer=" << layer << "=" << touch->GetReplicaNumber(0) << "\t"
  //  	    << "mod="   << module << "=" << touch->GetReplicaNumber(1) << "\t"
  //	    << "izplmin=" << iz  << "=" << touch->GetReplicaNumber(3) << std::endl; 
  //    int layer    = (touch->GetReplicaNumber(0));
  //  int module = (touch->GetReplicaNumber(1));
  //  int izplmin = (touch->GetReplicaNumber(3)); 

  return setDetUnitId (subdet, layer, module, iz, localpos, dz, bl1, tl1, h1);
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
uint32_t HGCSD::setDetUnitId (ForwardSubdetector &subdet, int &layer, int &module, int &iz, G4ThreeVector &pos, float &dz, float &bl1, float &tl1, float &h1)
{  
  return (numberingScheme ? numberingScheme->getUnitID(subdet, layer, module, iz, pos, dz, bl1, tl1, h1) : 0);
}

//
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
