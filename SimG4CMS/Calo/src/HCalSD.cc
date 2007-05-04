///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4Step.hh"
#include "G4Track.hh"

HCalSD::HCalSD(G4String name, const DDCompactView & cpv,
               edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, p, manager), numberingFromDDD(0), numberingScheme(0), 
  showerLibrary(0), hfshower(0) {

  //static SimpleConfigurable<bool>   on1(false, "HCalSD:UseBirkLaw");
  //static SimpleConfigurable<double> bk1(0.013, "HCalSD:BirkC1");
  //static SimpleConfigurable<double> bk2(9.6e-6,"HCalSD:BirkC2");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  //static SimpleConfigurable<bool> on2(true,"HCalSD:UseShowerLibrary");

  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HCalSD");
  useBirk    = m_HC.getParameter<bool>("UseBirkLaw");
  birk1      = m_HC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2      = m_HC.getParameter<double>("BirkC2")*(g/(MeV*cm2))*(g/(MeV*cm2));
  useShowerLibrary = m_HC.getParameter<bool>("UseShowerLibrary");
  useHF      = m_HC.getUntrackedParameter<bool>("UseHF",true);
  bool testNumber  = m_HC.getParameter<bool>("TestNumberingScheme");
  bool forTBH2     = m_HC.getUntrackedParameter<bool>("ForTBH2",false);

  LogDebug("HcalSim") << "***************************************************" 
		      << "\n"
		      << "*                                                 *"
		      << "\n"
		      << "* Constructing a HCalSD  with name " << name << "\n"
		      << "*                                                 *"
		      << "\n"
		      << "***************************************************";

  edm::LogInfo("HcalSim") << "HCalSD:: Use of HF code is set to " << useHF
			  << "\nUse of shower library is set to " 
			  << useShowerLibrary << "\n"
			  << "         Use of Birks law is set to      " 
			  << useBirk << "  with the two constants C1 = "
			  << birk1 << ", C2 = " << birk2;
  
  numberingFromDDD = new HcalNumberingFromDDD(name, cpv);
  HcalNumberingScheme* scheme;
  if (testNumber || forTBH2) 
    scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(forTBH2));
  else            
    scheme = new HcalNumberingScheme();
  setNumberingScheme(scheme);

  std::string attribute, value;
  if (useHF) {
    if (useShowerLibrary) showerLibrary = new HFShowerLibrary(name, cpv, p);
    else                  hfshower      = new HFShower(cpv,p);

    // HF volume names
    attribute = "Volume";
    value     = "HF";
    DDSpecificsFilter filter0;
    DDValue           ddv0(attribute,value,0);
    filter0.setCriteria(ddv0,DDSpecificsFilter::equals);
    DDFilteredView fv0(cpv);
    fv0.addFilter(filter0);
    if (fv0.firstChild()) hfNames = getNames(fv0);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
			    << " = " << value << ":";
    for (unsigned int i=0; i<hfNames.size(); i++)
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << hfNames[i];
  
    // HF Fibre volume names
    value     = "HFFibre";
    DDSpecificsFilter filter1;
    DDValue           ddv1(attribute,value,0);
    filter1.setCriteria(ddv1,DDSpecificsFilter::equals);
    DDFilteredView fv1(cpv);
    fv1.addFilter(filter1);
    fibreNames = getNames(fv1);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
			    << " = " << value << ":";
    for (unsigned int i=0; i<fibreNames.size(); i++)
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << fibreNames[i];
  }

  //Material list for HB/HE/HO sensitive detectors
  attribute = "ReadOutName";
  DDSpecificsFilter filter2;
  DDValue           ddv2(attribute,name,0);
  filter2.setCriteria(ddv2,DDSpecificsFilter::equals);
  DDFilteredView fv2(cpv);
  fv2.addFilter(filter2);
  bool dodet = fv2.firstChild();

  DDsvalues_type sv(fv2.mergedSpecifics());
  //Layer0 Weight
  layer0wt = getDDDArray("Layer0Wt",sv);
  edm::LogInfo("HcalSim") << "HCalSD: " << layer0wt.size() << " Layer0Wt";
  for (unsigned int it=0; it<layer0wt.size(); it++)
    edm::LogInfo("HcalSim") << "HCalSD: [" << it << "] " << layer0wt[it];

  while (dodet) {
    const DDLogicalPart & log = fv2.logicalPart();
    G4String namx = DDSplit(log.name()).first;
    if (!isItHF(namx) && !isItFibre(namx)) {
      namx = DDSplit(log.material().name()).first;
      bool notIn = true;
      for (unsigned int i=0; i<matNames.size(); i++)
	if (namx == matNames[i]) notIn = false;
      if (notIn) matNames.push_back(namx);
    }
    dodet = fv2.next();
  }

  edm::LogInfo("HcalSim") << "HCalSD: Material names for " << attribute 
			  << " = " << name << ":";
  for (unsigned int i=0; i<matNames.size(); i++)
    edm::LogInfo("HcalSim") << "HCalSD: (" << i << ") " << matNames[i];
}

HCalSD::~HCalSD() { 
  if (numberingFromDDD) delete numberingFromDDD;
  if (numberingScheme)  delete numberingScheme;
  if (showerLibrary)    delete showerLibrary;
  if (hfshower)         delete hfshower;
}

bool HCalSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  //  TimeMe t1( theHitTimer, false);
  
  
  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    G4String nameVolume = 
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetName();
    if (isItHF(nameVolume) || isItFibre(nameVolume)) {
      if (useShowerLibrary) {
	LogDebug("HcalSim") << "HCalSD: Starts shower library from " 
			    << nameVolume << " for Track "
			    << aStep->GetTrack()->GetTrackID() <<" ("
			    << aStep->GetTrack()->GetDefinition()->GetParticleName()
			    << ")";
	getFromLibrary(aStep);
      } else if (isItFibre(nameVolume)) {
	hitForFibre(aStep);
      }
    } else {
      getStepInfo(aStep);
      if (hitExists() == false && edepositEM+edepositHAD>0.) 
	createNewHit();
    }
    return true;
  }
} 

double HCalSD::getEnergyDeposit(G4Step* aStep) {

  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int depth = (touch->GetReplicaNumber(0))%10;
  int det   = ((touch->GetReplicaNumber(1))/1000)-3;
  if (depth==0 && (det==0 || det==1)) weight = layer0wt[det];
  double weight0 = weight;
  if (useBirk) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    if (isItScintillator(mat->GetName()))
      weight *= getAttenuation(aStep, birk1, birk2);
  }
  LogDebug("HcalSim") << "HCalSD: Detector " << det+3 << " Depth " << depth
		      << " weight " << weight0 << " " << weight;
  return weight*destep;
}

uint32_t HCalSD::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();

  int depth = (touch->GetReplicaNumber(0))%10 + 1;
  int lay   = (touch->GetReplicaNumber(0)/10)%100 + 1;
  int det   = (touch->GetReplicaNumber(1))/1000;

  return setDetUnitId (det, hitPoint, depth, lay);
}

void HCalSD::setNumberingScheme(HcalNumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogWarning("HcalSim") << "HCalSD: updates numbering scheme for " 
			       << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

uint32_t HCalSD::setDetUnitId (int det, G4ThreeVector pos, int depth, 
			       int lay=1) { 

  uint32_t id = 0;
  if (numberingFromDDD) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, 
								depth, lay);
    //get the ID
    if (numberingScheme) id = numberingScheme->getUnitID(tmp);
  }
  return id;
}

std::vector<double> HCalSD::getDDDArray(const std::string & str,
					const DDsvalues_type & sv) {

  LogDebug("HcalSim") << "HCalSD:getDDDArray called for " << str;
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HcalSim") << value;
    const vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HcalSim") << "HCalSD : # of " << str << " bins " << nval
			       << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HCalSD")
	<< "nval < 2 for array " << str << "\n";
    }
    
    return fvec;
  } else {
    edm::LogError("HcalSim") << "HCalSD :  cannot get array " << str;
    throw cms::Exception("Unknown", "HCalSD")
      << "cannot get array " << str << "\n";
  }
}

std::vector<G4String> HCalSD::getNames(DDFilteredView& fv) {

  std::vector<G4String> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart & log = fv.logicalPart();
    G4String namx = DDSplit(log.name()).first;
    bool ok = true;
    for (unsigned int i=0; i<tmp.size(); i++)
      if (namx == tmp[i]) ok = false;
    if (ok) tmp.push_back(namx);
    dodet = fv.next();
  }
  return tmp;
}

bool HCalSD::isItHF (G4String name) {

  std::vector<G4String>::const_iterator it = hfNames.begin();
  for (; it != hfNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

bool HCalSD::isItFibre (G4String name) {

  std::vector<G4String>::const_iterator it = fibreNames.begin();
  for (; it != fibreNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

bool HCalSD::isItScintillator (G4String name) {

  std::vector<G4String>::const_iterator it = matNames.begin();
  for (; it != matNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

void HCalSD::getFromLibrary (G4Step* aStep) {

  preStepPoint = aStep->GetPreStepPoint(); 
  theTrack     = aStep->GetTrack();   

  int nhit     = showerLibrary->getHits(aStep);
  int det      = 5;

  double etrack    = preStepPoint->GetKineticEnergy();
  int    primaryID = 0;
  if (etrack >= energyCut) {
    primaryID    = theTrack->GetTrackID();
  } else {
    primaryID    = theTrack->GetParentID();
    if (primaryID == 0) primaryID = theTrack->GetTrackID();
  }

  // Reset entry point for new primary
  posGlobal = preStepPoint->GetPosition();
  resetForNewPrimary(posGlobal, etrack);

  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  if (particleType == "e-" || particleType == "e+" ||
      particleType == "gamma" ) {
    edepositEM  = 1.*GeV; edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = 1.*GeV;
  }

  LogDebug("HcalSim") << "HCalSD: " << nhit << " hits for " << GetName() 
		      << " of " << primaryID << " with " << particleType 
		      << " of " << etrack/GeV << " GeV";

  for (int i=0; i<nhit; i++) {
    G4ThreeVector hitPoint = showerLibrary->getPosHit(i);
    int depth              = showerLibrary->getDepth(i);
    double time            = showerLibrary->getTSlice(i);
    unsigned int unitID    = setDetUnitId(det, hitPoint, depth);
    currentID.setID(unitID, time, primaryID);
   
    // check if it is in the same unit and timeslice as the previosus one
    if (currentID == previousID) {
      updateHit();
    } else {
      if (!checkHit()) createNewHit();
    }
  }

  //Now kill the current track
  if (nhit >= 0) {
    theTrack->SetTrackStatus(fStopAndKill);
  }
}

void HCalSD::hitForFibre (G4Step* aStep) {

  preStepPoint = aStep->GetPreStepPoint();
  theTrack     = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
    LogDebug("HcalSim") << "HCalSD: Problem with primaryID **** set by force "
			<< "to TkID **** " << theTrack->GetTrackID();
    primaryID = theTrack->GetTrackID();
  }

  int det   = (preStepPoint->GetTouchable()->GetReplicaNumber(1))/1000;
  int nHit  = hfshower->getHits(aStep);

  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  if (particleType == "e-" || particleType == "e+" ||
      particleType == "gamma" ) {
    edepositEM  = 1.*GeV; edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = 1.*GeV;
  }
 
  LogDebug("HcalSim") << "HCalSD: " << nHit << " hits for " << GetName() 
		      << " of " << primaryID << " with " << particleType 
		      << " of " << etrack/GeV << " GeV" << " in detector type "
		      << det;
 
  if (nHit > 0) {
    if (primaryID != previousID.trackID())
      resetForNewPrimary(preStepPoint->GetPosition(), etrack);
 
    G4ThreeVector hitPoint = preStepPoint->GetPosition();
    int           depth    = 
      (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10;
    unsigned int unitID    = setDetUnitId(det, hitPoint, depth);

    for (int i=0; i<nHit; i++) {
      double time            = hfshower->getTSlice(i);
      currentID.setID(unitID, time, primaryID);

      // check if it is in the same unit and timeslice as the previosus one
      if (currentID == previousID) {
        updateHit();
      } else {
        posGlobal = preStepPoint->GetPosition();
        if (!checkHit()) createNewHit();
      }
    }
  }

}
