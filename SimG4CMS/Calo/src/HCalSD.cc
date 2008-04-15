///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"

#include <iostream>
#include <fstream>
#include <iomanip>

HCalSD::HCalSD(G4String name, const DDCompactView & cpv,
	       SensitiveDetectorCatalog & clg, 
               edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager), numberingFromDDD(0), numberingScheme(0), 
  showerLibrary(0), hfshower(0), showerParam(0), showerPMT(0) {

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
  useParam         = m_HC.getParameter<bool>("UseParametrize");
  bool testNumber  = m_HC.getParameter<bool>("TestNumberingScheme");
  usePMTHit        = m_HC.getParameter<bool>("UsePMTHits");
  betaThr          = m_HC.getParameter<double>("BetaThreshold");
  useHF            = m_HC.getUntrackedParameter<bool>("UseHF",true);
  bool forTBH2     = m_HC.getUntrackedParameter<bool>("ForTBH2",false);
  useLayerWt       = m_HC.getUntrackedParameter<bool>("UseLayerWt",false);
  std::string file = m_HC.getUntrackedParameter<std::string>("WtFile","None");

  LogDebug("HcalSim") << "***************************************************" 
		      << "\n"
		      << "*                                                 *"
		      << "\n"
		      << "* Constructing a HCalSD  with name " << name << "\n"
		      << "*                                                 *"
		      << "\n"
		      << "***************************************************";

  edm::LogInfo("HcalSim") << "HCalSD:: Use of HF code is set to " << useHF
			  << "\nUse of shower parametrization set to "
			  << useParam << "\nUse of shower library is set to " 
			  << useShowerLibrary << "\nUse PMT Hit is set to "
			  << usePMTHit << " with beta Threshold "<< betaThr
			  << "\n         Use of Birks law is set to      " 
			  << useBirk << "  with the two constants C1 = "
			  << birk1 << ", C2 = " << birk2;
  
  edm::LogInfo("HcalSim") << "HCalSD:: Suppression Flag " << suppressHeavy
			  << " protons below " << kmaxProton << " MeV,"
			  << " neutrons below " << kmaxNeutron << " MeV and"
			  << " ions below " << kmaxIon << " MeV";

  numberingFromDDD = new HcalNumberingFromDDD(name, cpv);
  HcalNumberingScheme* scheme;
  if (testNumber || forTBH2) 
    scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(forTBH2));
  else            
    scheme = new HcalNumberingScheme();
  setNumberingScheme(scheme);

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  G4LogicalVolume* lv;
  std::string attribute, value;
  if (useHF) {
    if (useParam) showerParam = new HFShowerParam(name, cpv, p);
    else {
      if (useShowerLibrary) showerLibrary = new HFShowerLibrary(name, cpv, p);
      hfshower  = new HFShower(name, cpv, p);
    }

    // HF volume names
    attribute = "Volume";
    value     = "HF";
    DDSpecificsFilter filter0;
    DDValue           ddv0(attribute,value,0);
    filter0.setCriteria(ddv0,DDSpecificsFilter::equals);
    DDFilteredView fv0(cpv);
    fv0.addFilter(filter0);
    hfNames = getNames(fv0);
    fv0.firstChild();
    DDsvalues_type sv0(fv0.mergedSpecifics());
    std::vector<double> temp =  getDDDArray("Levels",sv0);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
			    << " = " << value << " has " << hfNames.size()
			    << " elements";
    for (unsigned int i=0; i<hfNames.size(); i++) {
      G4String namv = hfNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
	if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      hfLV.push_back(lv);
      int level = static_cast<int>(temp[i]);
      hfLevels.push_back(level);
      edm::LogInfo("HcalSim") << "HCalSD:  HF[" << i << "] = " << hfNames[i]
			      << " LV " << hfLV[i] << " at level " 
			      << hfLevels[i];
    }
  
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
    for (unsigned int i=0; i<fibreNames.size(); i++) {
      G4String namv = fibreNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
	if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      fibreLV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << fibreNames[i]
			      << " LV " << fibreLV[i];
    }
  
    // HF PMT volume names
    value     = "HFPMT";
    DDSpecificsFilter filter3;
    DDValue           ddv3(attribute,value,0);
    filter3.setCriteria(ddv3,DDSpecificsFilter::equals);
    DDFilteredView fv3(cpv);
    fv3.addFilter(filter3);
    pmtNames = getNames(fv3);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute
			    << " = " << value << " have " << pmtNames.size()
			    << " entries";
    for (unsigned int i=0; i<pmtNames.size(); i++) {
      G4String namv = pmtNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
	if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      pmtLV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << pmtNames[i]
			      << " LV " << pmtLV[i];
    }
    if (pmtNames.size() > 0) showerPMT = new HFShowerPMT (name, cpv, p);
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

  const G4MaterialTable * matTab = G4Material::GetMaterialTable();
  std::vector<G4Material*>::const_iterator matite;
  while (dodet) {
    const DDLogicalPart & log = fv2.logicalPart();
    G4String namx = DDSplit(log.name()).first;
    if (!isItHF(namx) && !isItFibre(namx)) {
      namx = DDSplit(log.material().name()).first;
      bool notIn = true;
      for (unsigned int i=0; i<matNames.size(); i++)
	if (namx == matNames[i]) notIn = false;
      if (notIn) {
	matNames.push_back(namx);
	G4Material* mat;
	for (matite = matTab->begin(); matite != matTab->end(); matite++) 
	  if ((*matite)->GetName() == namx) {
	    mat = (*matite);
	    break;
	  }
	materials.push_back(mat);
      }
    }
    dodet = fv2.next();
  }

  edm::LogInfo("HcalSim") << "HCalSD: Material names for " << attribute 
			  << " = " << name << ":";
  for (unsigned int i=0; i<matNames.size(); i++)
    edm::LogInfo("HcalSim") << "HCalSD: (" << i << ") " << matNames[i]
			    << " pointer " << materials[i];

  mumPDG = mupPDG = 0;
  
  if (useLayerWt) readWeightFromFile(file);
}

HCalSD::~HCalSD() { 
  if (numberingFromDDD) delete numberingFromDDD;
  if (numberingScheme)  delete numberingScheme;
  if (showerLibrary)    delete showerLibrary;
  if (hfshower)         delete hfshower;
  if (showerParam)      delete showerParam;
  if (showerPMT)        delete showerPMT;
}

bool HCalSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {
  //  TimeMe t1( theHitTimer, false);
  
  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    G4LogicalVolume* lv = 
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
    G4String nameVolume = lv->GetName();
    if (isItHF(aStep)) {
      G4int parCode =aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
      if (useParam) {
        LogDebug("HcalSim") << "HCalSD: Hit from parametrization in " 
                            << nameVolume << " for Track " 
                            << aStep->GetTrack()->GetTrackID()
                            <<" (" << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
        getFromParam(aStep);
      } else {
        bool notaMuon = true;
        if (parCode == mupPDG || parCode == mumPDG ) notaMuon = false;
        if (useShowerLibrary && notaMuon) {
          LogDebug("HcalSim") << "HCalSD: Starts shower library from " 
                              << nameVolume << " for Track " 
                              << aStep->GetTrack()->GetTrackID()
                              <<" (" << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
          getFromLibrary(aStep);
        } else if (isItFibre(lv)) {
          LogDebug("HcalSim") << "HCalSD: Hit at Fibre in " << nameVolume 
                              << " for Track " 
                              << aStep->GetTrack()->GetTrackID()
                              <<" ("  << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
          hitForFibre(aStep);
        }
      }
    } else if (isItPMT(lv)) {
      LogDebug("HcalSim") << "HCalSD: Hit from PMT parametrization from " 
                          <<  nameVolume << " for Track " 
                          << aStep->GetTrack()->GetTrackID() << " ("
                          << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
      if (usePMTHit && showerPMT) getHitPMT(aStep);
    } else {
      LogDebug("HcalSim") << "HCalSD: Hit from standard path from " 
                          <<  nameVolume << " for Track " 
                          << aStep->GetTrack()->GetTrackID() << " ("
                          << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
      if (getStepInfo(aStep)) {
        if (hitExists() == false && edepositEM+edepositHAD>0.) 
          currentHit = createNewHit();
      }
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
  if (useLayerWt) {
    int lay   = (touch->GetReplicaNumber(0)/10)%100 + 1;
    G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
    weight = layerWeight(det+3, hitPoint, depth, lay);
  }

  if (suppressHeavy) {
    G4Track* theTrack = aStep->GetTrack();
    TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) { // Only secondary particles
	double ke = theTrack->GetKineticEnergy()/MeV;
	if (((pdg/1000000000 == 1 && ((pdg/10000)%100) > 0 &&
	      ((pdg/10)%100) > 0)) && (ke<kmaxIon)) weight = 0;
	if ((pdg == 2212) && (ke < kmaxProton))     weight = 0;
	if ((pdg == 2112) && (ke < kmaxNeutron))    weight = 0;
	if (weight == 0) {
	  LogDebug("HcalSim") << "Ignore Track " << theTrack->GetTrackID()
			      << " Type " << theTrack->GetDefinition()->GetParticleName()
			      << " Kinetic Energy " << ke << " MeV";
	}
      }
    }
  }

  double weight0 = weight;
  if (useBirk) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    if (isItScintillator(mat))
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
    edm::LogInfo("HcalSim") << "HCalSD: updates numbering scheme for " 
			    << GetName() << "\n";
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void HCalSD::initRun() {

  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String          particleName;
  mumPDG = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
  LogDebug("HcalSim") << "HCalSD: Particle code for mu- = " << mumPDG
		      << " for mu+ = " << mupPDG;

  if (showerLibrary) showerLibrary->initRun(theParticleTable);
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
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
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

bool HCalSD::isItHF (G4Step * aStep) {

  const G4VTouchable* touch    = aStep->GetPreStepPoint()->GetTouchable();
  int levels = ((touch->GetHistoryDepth())+1);
  for (unsigned int it=0; it < hfNames.size(); it++) {
    if (levels >= hfLevels[it]) {
      G4LogicalVolume* lv = touch->GetVolume(levels-hfLevels[it])->GetLogicalVolume();
      if (lv == hfLV[it]) return true;
    }
  }
  return false;
}

bool HCalSD::isItHF (G4String name) {

  std::vector<G4String>::const_iterator it = hfNames.begin();
  for (; it != hfNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

bool HCalSD::isItFibre (G4LogicalVolume* lv) {

  std::vector<G4LogicalVolume*>::const_iterator ite = fibreLV.begin();
  for (; ite != fibreLV.end(); ite++) 
    if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItFibre (G4String name) {

  std::vector<G4String>::const_iterator it = fibreNames.begin();
  for (; it != fibreNames.end(); it++) 
    if (name == *it) return true;
  return false;
}

bool HCalSD::isItPMT (G4LogicalVolume* lv) {

  std::vector<G4LogicalVolume*>::const_iterator ite = pmtLV.begin();
  for (; ite != pmtLV.end(); ite++) 
    if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItScintillator (G4Material* mat) {

  std::vector<G4Material*>::const_iterator ite = materials.begin();
  for (; ite != materials.end(); ite++) 
    if (mat == *ite) return true;
  return false;
}

void HCalSD::getFromLibrary (G4Step* aStep) {

  preStepPoint  = aStep->GetPreStepPoint(); 
  theTrack      = aStep->GetTrack();   
  int det       = 5;
  bool ok;

  std::vector<HFShowerLibrary::Hit> hits = showerLibrary->getHits(aStep, ok);

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
  //  int primaryID = setTrackID(aStep);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if ( particleCode == emPDG || particleCode == epPDG || particleCode == gammaPDG ) {
    edepositEM  = 1.*GeV; edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = 1.*GeV;
  }

  LogDebug("HcalSim") << "HCalSD::getFromLibrary " <<hits.size() <<" hits for "
		      << GetName() << " of " << primaryID << " with " 
		      << theTrack->GetDefinition()->GetParticleName() << " of " 
		      << preStepPoint->GetKineticEnergy()/GeV << " GeV";

  for (unsigned int i=0; i<hits.size(); i++) {
    G4ThreeVector hitPoint = hits[i].position;
    int depth              = hits[i].depth;
    double time            = hits[i].time;
    unsigned int unitID    = setDetUnitId(det, hitPoint, depth);
    currentID.setID(unitID, time, primaryID, 0);
   
    // check if it is in the same unit and timeslice as the previous one
    if (currentID == previousID) {
      updateHit(currentHit);
    } else {
      if (!checkHit()) currentHit = createNewHit();
    }
  }

  //Now kill the current track
  if (ok) {
    theTrack->SetTrackStatus(fStopAndKill);
  }
}

void HCalSD::hitForFibre (G4Step* aStep) {

  preStepPoint  = aStep->GetPreStepPoint();
  theTrack      = aStep->GetTrack();
  int primaryID = setTrackID(aStep);

  int det   = 5;
  int nHit  = hfshower->getHits(aStep);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if ( particleCode == emPDG || particleCode == epPDG || particleCode == gammaPDG ) {
    edepositEM  = 1.*GeV; edepositHAD = 0.;
  } else {
    edepositEM  = 0.; edepositHAD = 1.*GeV;
  }
 
  LogDebug("HcalSim") << "HCalSD::hitForFibre " << nHit << " hits for " 
		      << GetName() << " of " << primaryID << " with " 
		      << theTrack->GetDefinition()->GetParticleName() << " of " 
		      << preStepPoint->GetKineticEnergy()/GeV 
		      << " GeV in detector type " << det;
 
  if (nHit > 0) {

    G4ThreeVector hitPoint = preStepPoint->GetPosition();
    int           depth    = 
      (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10;
    unsigned int unitID    = setDetUnitId(det, hitPoint, depth);

    for (int i=0; i<nHit; i++) {
      double time            = hfshower->getTSlice(i);
      currentID.setID(unitID, time, primaryID, 0);

      // check if it is in the same unit and timeslice as the previous one
      if (currentID == previousID) {
        updateHit(currentHit);
      } else {
        posGlobal = preStepPoint->GetPosition();
        if (!checkHit()) currentHit = createNewHit();
      }
    }
  }

}

void HCalSD::getFromParam (G4Step* aStep) {

  std::vector<HFShowerParam::Hit> hits = showerParam->getHits(aStep);
  int nHit = static_cast<int>(hits.size());

  if (nHit > 0) {
    preStepPoint  = aStep->GetPreStepPoint();
    int primaryID = setTrackID(aStep);
   
    int det   = 5;
    LogDebug("HcalSim") << "HCalSD::getFromParam " << nHit << " hits for " 
			<< GetName() << " of " << primaryID << " with " 
			<<  aStep->GetTrack()->GetDefinition()->GetParticleName()
			<< " of " << preStepPoint->GetKineticEnergy()/GeV 
			<< " GeV in detector type " << det;

    for (int i = 0; i<nHit; i++) {
      G4ThreeVector hitPoint = hits[i].position;
      int depth              = hits[i].depth;
      double time            = hits[i].time;
      unsigned int unitID    = setDetUnitId(det, hitPoint, depth);
      currentID.setID(unitID, time, primaryID, 0);
      edepositEM             = hits[i].edep*GeV; 
      edepositHAD            = 0;

      // check if it is in the same unit and timeslice as the previous one
      if (currentID == previousID) {
        updateHit(currentHit);
      } else {
        posGlobal = preStepPoint->GetPosition();
        if (!checkHit()) currentHit = createNewHit();
      }
    }
  }
}

void HCalSD::getHitPMT (G4Step* aStep) {

  preStepPoint = aStep->GetPreStepPoint();
  theTrack     = aStep->GetTrack();
  double edep  = showerPMT->getHits(aStep);

  if (edep > 0) {
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

    //
    int    det      = static_cast<int>(HcalForward);
    G4ThreeVector hitPoint = preStepPoint->GetPosition();   
    double rr       = (hitPoint.x()*hitPoint.x() + hitPoint.y()*hitPoint.y());
    double phi      = (rr == 0. ? 0. :atan2(hitPoint.y(),hitPoint.x()));
    double etaR     = showerPMT->getRadius();
    int depth       = 1;
    if (etaR < 0) {
      depth         = 2;
      etaR          =-etaR;
    }
    if (hitPoint.z() < 0) etaR =-etaR;
    LogDebug("HcalSim") << "HCalSD::Hit for Detector " << det << " etaR "
			<< etaR << " phi " << phi/deg << " depth " << depth;

    double time = (aStep->GetPostStepPoint()->GetGlobalTime());
    uint32_t unitID = 0;
    if (numberingFromDDD) {
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det,etaR,phi,
								  depth,1);
      if (numberingScheme) unitID = numberingScheme->getUnitID(tmp);
    }
    currentID.setID(unitID, time, primaryID, 1);

    double beta = preStepPoint->GetBeta();
    if (beta > betaThr) {
      edepositEM  = edep*GeV; edepositHAD = aStep->GetTotalEnergyDeposit();
    } else {
      edepositEM  = 0.; edepositHAD = aStep->GetTotalEnergyDeposit();
    }
    LogDebug("HcalSim") << "HCalSD::getHitPMT 1 hit for " << GetName() 
                        << " of " << primaryID << " with " 
			<< theTrack->GetDefinition()->GetParticleName()
                        << " of " << preStepPoint->GetKineticEnergy()/GeV 
                        << " GeV with velocity " << beta << " UnitID "
                        << std::hex << unitID << std::dec;

    // check if it is in the same unit and timeslice as the previous one
    if (currentID == previousID) {
      updateHit(currentHit);
    } else {
      if (!checkHit()) currentHit = createNewHit();
    }
  }
}

int HCalSD::setTrackID (G4Step* aStep) {

  theTrack     = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
    LogDebug("HcalSim") << "HCalSD: Problem with primaryID **** set by force "
			<< "to TkID **** " << theTrack->GetTrackID();
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID())
    resetForNewPrimary(preStepPoint->GetPosition(), etrack);

  return primaryID;
}

void HCalSD::readWeightFromFile(std::string fName) {

  std::ifstream infile;
  int entry=0;
  infile.open(fName.c_str(), std::ios::in);
  if (infile) {
    int    det, zside, etaR, phi, lay;
    double wt;
    while (infile >> det >> zside >> etaR >> phi >> lay >> wt) {
      uint32_t id = HcalTestNumbering::packHcalIndex(det,zside,1,etaR,phi,lay);
      layerWeights.insert(std::pair<uint32_t,double>(id,wt));
      entry++;
      LogDebug("HcalSim") << "HCalSD::readWeightFromFile:Entry " << entry
			  << " ID " << std::hex << id << std::dec << " ("
			  << det << "/" << zside << "/1/" << etaR << "/"
			  << phi << "/" << lay << ") Weight " << wt;
    }
    infile.close();
  }
  edm::LogInfo("HcalSim") << "HCalSD::readWeightFromFile: reads " << entry
			  << " weights from " << fName;
  if (entry <= 0) useLayerWt = false;
}

double HCalSD::layerWeight(int det, G4ThreeVector pos, int depth, int lay) { 

  double wt = 1.;
  if (numberingFromDDD) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, 
								depth, lay);
    uint32_t id = HcalTestNumbering::packHcalIndex(tmp.subdet, tmp.zside,
						   1, tmp.etaR, tmp.phis,
						   tmp.lay);
    std::map<uint32_t,double>::const_iterator ite = layerWeights.find(id);
    if (ite != layerWeights.end()) wt = ite->second;
    LogDebug("HcalSim") << "HCalSD::layerWeight: ID " << std::hex << id 
			<< std::dec << " (" << tmp.subdet << "/"  << tmp.zside 
			<< "/1/" << tmp.etaR << "/" << tmp.phis << "/" 
			<< tmp.lay << ") Weight " << wt;
  }
  return wt;
}
