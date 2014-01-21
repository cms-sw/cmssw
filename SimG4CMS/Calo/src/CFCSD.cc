///////////////////////////////////////////////////////////////////////////////
// File: CFCSD.cc
// Description: Sensitive Detector class for Combined Forward Calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CFCSD.h"
#include "SimG4CMS/Calo/interface/CFCNumberingScheme.h"
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
#include "Randomize.hh"

#include <iostream>
#include <fstream>
#include <iomanip>

#define DebugLog

CFCSD::CFCSD(G4String name, const DDCompactView & cpv,
	     SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         p.getParameter<edm::ParameterSet>("CFCSD").getParameter<int>("TimeSliceUnit"),
         p.getParameter<edm::ParameterSet>("CFCSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme(0), showerLibrary(0) {

  edm::ParameterSet m_CFC = p.getParameter<edm::ParameterSet>("CFCSD");
  eminHit          = m_CFC.getParameter<double>("EminHit")*MeV;
  cFibre           = c_light*(m_CFC.getParameter<double>("CFibre"));
  applyFidCut      = m_CFC.getParameter<bool>("ApplyFiducialCut");

#ifdef DebugLog
  LogDebug("CFCSim") << "**************************************************" 
                      << "\n"
                      << "*                                                *"
                      << "\n"
                      << "* Constructing a CFCSD  with name " << name << "\n"
                      << "*                                                *"
                      << "\n"
                      << "**************************************************";
#endif
  edm::LogInfo("CFCSim") << "CFCSD:: Threshold for storing hits: " << eminHit
			 << " Speed of light in fibre " << cFibre << " m/ns"
			 << " Application of Fiducial Cut " << applyFidCut;

  std::string attribute, value;
  // Constants for Numbering Scheme, shower, attenuation length
  attribute = "Volume";
  value     = "CFC";
  DDSpecificsFilter filter0;
  DDValue           ddv0(attribute, value, 0);
  filter0.setCriteria(ddv0, DDSpecificsFilter::equals);
  DDFilteredView fv0(cpv);
  fv0.addFilter(filter0);
  fv0.firstChild();
  DDsvalues_type sv0(fv0.mergedSpecifics());

  std::vector<double> rv = getDDDArray("RadiusTable",sv0);
  std::vector<double> xv = getDDDArray("XcellSize",sv0);
  std::vector<double> yv = getDDDArray("YcellSize",sv0);
  numberingScheme = new CFCNumberingScheme(rv,xv,yv);

  attL    = getDDDArray("AttenuationLength",sv0);
  lambLim = getDDDArray("LambdaLimit",sv0);
  gpar    = getDDDArray("GeomParCFC",sv0);
  showerLibrary = new CFCShowerLibrary(p, gpar);
  nBinAtt = (int)(attL.size());
  edm::LogInfo("CFCSim") << "CFCSD: " << nBinAtt << " attenuation lengths "
			 << "for lambda in the range " << lambLim[0] << ":"
			 << lambLim[1];
  for (int i=0; i<nBinAtt; ++i) 
    edm::LogInfo("CFCSim") << "attL[" << i << "] = " << attL[i];
  edm::LogInfo("CFCSim") << "CFCSD: " << gpar.size() << " geometry parameters";
  for (unsigned int i=0; i<gpar.size(); ++i) 
    edm::LogInfo("CFCSim") << "gpar[" << i << "] = " << gpar[i];
}

CFCSD::~CFCSD() { 

  if (numberingScheme)  delete numberingScheme;
  if (showerLibrary)    delete showerLibrary;
}

bool CFCSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
    bool notaMuon = true;
    if (parCode == mupPDG || parCode == mumPDG ) notaMuon = false;
    if (notaMuon) {
#ifdef DebugLog
      G4LogicalVolume* lv =
	aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
      edm::LogInfo("CFCSim") << "CFCSD: Starts shower library from " 
			     << lv->GetName() << " for Track " 
			     << aStep->GetTrack()->GetTrackID() << " ("
			     << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      getFromLibrary(aStep);
    } else {
#ifdef DebugLog
      G4LogicalVolume* lv =
	aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
      edm::LogInfo("CFCSim") << "CFCSD: Hit from standard path from " 
			     << lv->GetName() << " for Track " 
			     << aStep->GetTrack()->GetTrackID() << " ("
			     << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      if (getStepInfo(aStep)) {
        if (hitExists() == false && edepositEM+edepositHAD>0.) currentHit = createNewHit();
      }
    }
    return true;
  }
} 

double CFCSD::getEnergyDeposit(G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
  return destep;
}

uint32_t CFCSD::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();

  int module = (touch->GetReplicaNumber(1));
  G4ThreeVector localpos = touch->GetHistory()->GetTopTransform().TransformPoint(hitPoint);
  int iz     = (hitPoint.z() > 0) ? 1 : -1;

  return setDetUnitId (module, localpos, iz, 0);
}

void CFCSD::initRun() {
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String          particleName;
  mumPDG = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
#ifdef DebugLog
  LogDebug("CFCSim") << "CFCSD: Particle code for mu- = " << mumPDG
		     << " for mu+ = " << mupPDG;
#endif
  if (showerLibrary) showerLibrary->initRun(theParticleTable);
}

bool CFCSD::filterHit(CaloG4Hit* aHit, double time) {
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > eminHit));
}


uint32_t CFCSD::setDetUnitId (int nmod, G4ThreeVector pos, int iz, int type){
  uint32_t id = 0;
  //get the ID
  if (numberingScheme) id = numberingScheme->getUnitID(pos, iz, nmod, type);
  return id;
}

std::vector<double> CFCSD::getDDDArray(const std::string & str,
                                        const DDsvalues_type & sv) {
#ifdef DebugLog
  LogDebug("CFCSim") << "CFCSD:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("CFCSim") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("CFCSim") << "CFCSD : # of " << str << " bins " << nval
			      << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "CFCSD") << "nval < 2 for array " << str << "\n";
    }
    
    return fvec;
  } else {
    edm::LogError("CFCSim") << "CFCSD :  cannot get array " << str;
    throw cms::Exception("Unknown", "CFCSD") << "cannot get array " << str << "\n";
  }
}

bool CFCSD::isItinFidVolume (G4ThreeVector& hitPoint) {
  bool flag = true;
  if (applyFidCut) {
    // Take a decision of selecting/rejecting based on local position
    if (hitPoint.z() < gpar[0] || hitPoint.z() > gpar[1]) flag = false;
  }
#ifdef DebugLog
    edm::LogInfo("CFCSim") << "CFCSD::isItinFidVolume: point " << hitPoint
			   << " return flag " << flag;
#endif
  return flag;
}

void CFCSD::getFromLibrary (G4Step* aStep) {
  preStepPoint  = aStep->GetPreStepPoint(); 
  theTrack      = aStep->GetTrack();   
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  int module = (touch->GetReplicaNumber(1));
  int iz     = (preStepPoint->GetPosition().z() > 0) ? 1 : -1;
  bool ok;

  std::vector<CFCShowerLibrary::Hit> hits = showerLibrary->getHits(aStep, ok);

  double etrack    = preStepPoint->GetKineticEnergy();
  int    primaryID = setTrackID(aStep);

  // Reset entry point for new primary
  posGlobal = preStepPoint->GetPosition();
  resetForNewPrimary(posGlobal, etrack);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (particleCode==emPDG || particleCode==epPDG || particleCode==gammaPDG) {
    edepositEM  = 1.*GeV;
    edepositHAD = 0.;
  } else {
    edepositEM  = 0.;
    edepositHAD = 1.*GeV;
  }
#ifdef DebugLog
  edm::LogInfo("CFCSim") << "CFCSD::getFromLibrary " <<hits.size() 
			 << " hits for " << GetName() << " of " << primaryID 
			 << " with " << theTrack->GetDefinition()->GetParticleName() 
			 << " of " << preStepPoint->GetKineticEnergy()/GeV << " GeV";
#endif
  for (unsigned int i=0; i<hits.size(); ++i) {
    G4ThreeVector hitPoint = hits[i].position;
    unsigned int  unitID   = setDetUnitId(module, hitPoint, iz, hits[i].type);
    double        zv       = fiberL(hitPoint);
    double        att      = attLength(hits[i].lambda);
    double        rn       = G4UniformRand();
    if (isItinFidVolume (hitPoint) && unitID > 0 && rn <= exp(-att*zv)) {
      double time          = hits[i].time + tShift(hitPoint);
      currentID.setID(unitID, time, primaryID, 0);
   
      // check if it is in the same unit and timeslice as the previous one
      if (currentID == previousID) {
	updateHit(currentHit);
      } else {
	if (!checkHit()) currentHit = createNewHit();
      }
    }
  }

  //Now kill the current track
  if (ok) {
    theTrack->SetTrackStatus(fStopAndKill);
    G4TrackVector tv = *(aStep->GetSecondary());
    for (unsigned int kk=0; kk<tv.size(); ++kk)
      if (tv[kk]->GetVolume() == preStepPoint->GetPhysicalVolume())
        tv[kk]->SetTrackStatus(fStopAndKill);
  }
}

int CFCSD::setTrackID (G4Step* aStep) {

  theTrack         = aStep->GetTrack();
  double etrack    = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int    primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
#ifdef DebugLog
    edm::LogInfo("CFCSim") << "CFCSD: Problem with primaryID **** set by "
			   << "force to TkID **** " <<theTrack->GetTrackID();
#endif
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID())
    resetForNewPrimary(preStepPoint->GetPosition(), etrack);

  return primaryID;
}

double CFCSD::attLength(double lambda) {

  int i = int(nBinAtt*(lambda - lambLim[0])/(lambLim[1]-lambLim[0]));
  int j =i;
  if (i >= nBinAtt) j = nBinAtt-1;
  else if (i < 0)   j = 0;
  double att = attL[j];
#ifdef DebugLog
  edm::LogInfo("CFCSim") << "CFCSD::attLength for Lambda " << lambda
			 << " index " << i  << ":" << j << " Att. Length " 
			 << att;
#endif
  return att;
}

double CFCSD::tShift(G4ThreeVector point) {

  double zFibre = fiberL(point);
  double time   = zFibre/cFibre;
#ifdef DebugLog
  edm::LogInfo("CFCSim") << "HFFibre::tShift for point " << point
			 << " (traversed length = " << zFibre/cm  << " cm) = " 
			 << time/ns << " ns";
#endif
  return time;
}

double CFCSD::fiberL(G4ThreeVector point) {

  double zFibre = gpar[1]-point.z();
  if (zFibre < 0) zFibre = 0;
#ifdef DebugLog
  edm::LogInfo("CFCSim") << "HFFibre::fiberL for point " << point << " = "
			 << zFibre/cm  << " cm";
#endif
  return zFibre;
}
