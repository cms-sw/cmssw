// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB04Analysis
//
// Implementation:
//     Main analysis class for Hcal Test Beam 2004 Analysis
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: HcalTB04Analysis.cc,v 1.11 2011/10/11 20:40:20 sunanda Exp $
//
  
// system include files
#include <cmath>
#include <iostream>
#include <iomanip>

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04Analysis.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTBNumberingScheme.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB04XtalNumberingScheme.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//
// constructors and destructor
//

HcalTB04Analysis::HcalTB04Analysis(const edm::ParameterSet &p): myQie(0),
								histo(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("HcalTB04Analysis");
  hcalOnly       = m_Anal.getParameter<bool>("HcalOnly");
  mode           = m_Anal.getParameter<int>("Mode");
  type           = m_Anal.getParameter<int>("Type");
  ecalNoise      = m_Anal.getParameter<double>("EcalNoise");
  scaleHB0       = m_Anal.getParameter<double>("ScaleHB0");
  scaleHB16      = m_Anal.getParameter<double>("ScaleHB16");
  scaleHO        = m_Anal.getParameter<double>("ScaleHO");
  scaleHE0       = m_Anal.getParameter<double>("ScaleHE0");
  names          = m_Anal.getParameter<std::vector<std::string> >("Names");
  beamOffset     =-m_Anal.getParameter<double>("BeamPosition")*cm;
  double fMinEta = m_Anal.getParameter<double>("MinEta");
  double fMaxEta = m_Anal.getParameter<double>("MaxEta");
  double fMinPhi = m_Anal.getParameter<double>("MinPhi");
  double fMaxPhi = m_Anal.getParameter<double>("MaxPhi");
  double beamEta = (fMaxEta+fMinEta)/2.;
  double beamPhi = (fMaxPhi+fMinPhi)/2.;
  double beamThet= 2*atan(exp(-beamEta));
  if (beamPhi < 0) beamPhi += twopi;
  iceta          = (int)(beamEta/0.087) + 1;
  icphi          = (int)(fabs(beamPhi)/0.087) + 5;
  if (icphi > 72) icphi -= 73;

  produces<PHcalTB04Info>();

  beamline_RM = new G4RotationMatrix;
  beamline_RM->rotateZ(-beamPhi);
  beamline_RM->rotateY(-beamThet);
 
  edm::LogInfo("HcalTBSim") << "HcalTB04:: Initialised as observer of BeginOf"
			    << "Job/BeginOfRun/BeginOfEvent/G4Step/EndOfEvent"
			    << " with Parameter values:\n \thcalOnly = " 
			    << hcalOnly << "\tecalNoise = " << ecalNoise
			    << "\n\tMode = " << mode << " (0: HB2 Standard; "
			    << "1:HB2 Segmented)" << "\tType = " << type 
			    << " (0: HB; 1 HE; 2 HB+HE)\n\tbeamOffset = " 
			    << beamOffset << "\ticeta = " << iceta 
			    << "\ticphi = " << icphi << "\n\tbeamline_RM = "
			    << *beamline_RM;

  init();

  myQie  = new HcalQie(p);
  histo  = new HcalTB04Histo(m_Anal);
} 
   
HcalTB04Analysis::~HcalTB04Analysis() {

  edm::LogInfo("HcalTBSim") << "\n -------->  Total number of selected entries"
			    << " : " << count << "\nPointers:: QIE " << myQie
			    << " Histo " << histo;
  if (myQie)   {
    delete myQie;
    myQie  = 0;
  }
  if (histo)   {
    delete histo;
    histo  = 0;
  }
}

//
// member functions
//

void HcalTB04Analysis::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<PHcalTB04Info> product(new PHcalTB04Info);
  fillEvent(*product);
  e.put(product);
}

void HcalTB04Analysis::init() {

  idTower = HcalTBNumberingScheme::getUnitIDs(type, mode);
  nTower  = idTower.size();
  edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Save information from " 
			    << nTower << " HCal towers";
  idHcal.reserve(nTower);
  for (int i=0; i<nTower; i++) {
    int id = unitID(idTower[i]);
    idHcal.push_back(id);
    LogDebug("HcalTBSim") << "\tTower[" << i << "] Original " << std::hex
			  << idTower[i] << " Stored " << idHcal[i] << std::dec;
  }

  if (!hcalOnly) {
    int  det = 10;
    uint32_t id1;
    nCrystal = 0;
    for (int lay=1; lay<8; lay++) {
      for (int icr=1; icr<8; icr++) {
	id1    = HcalTestNumbering::packHcalIndex(det,0,1,icr,lay,1);
	int id = unitID(id1);
	idEcal.push_back(id1);
	idXtal.push_back(id);
	nCrystal++;
      }
    }
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Save information from " 
			      << nCrystal << " ECal Crystals";
    for (int i=0; i<nCrystal; i++) {
      LogDebug("HcalTBSim") << "\tCrystal[" << i << "] Original " << std::hex
			    << idEcal[i] << " Stored " << idXtal[i] <<std::dec;
    }
  }
  // Profile vectors
  eseta.reserve(5);
  eqeta.reserve(5); 
  esphi.reserve(3);
  eqphi.reserve(3);
  eslay.reserve(20);
  eqlay.reserve(20);
  for (int i=0; i<5; i++) {
    eseta.push_back(0.);
    eqeta.push_back(0.);
  }
  for (int i=0; i<3; i++) {
    esphi.push_back(0.);
    eqphi.push_back(0.);
  }
  for (int i=0; i<20; i++) {
    eslay.push_back(0.);
    eqlay.push_back(0.);
  }

  // counter 
  count = 0;
  evNum = 0;
  clear();
}

void HcalTB04Analysis::update(const BeginOfRun * run) {

  int irun = (*run)()->GetRunID();
  edm::LogInfo("HcalTBSim") <<" =====> Begin of Run = " << irun;
 
  G4SDManager* sd     = G4SDManager::GetSDMpointerIfExist();
  if (sd != 0) {
    std::string  sdname = names[0];
    G4VSensitiveDetector* aSD = sd->FindSensitiveDetector(sdname);
    if (aSD==0) {
      edm::LogWarning("HcalTBSim") << "HcalTB04Analysis::beginOfRun: No SD"
				   << " with name " << sdname << " in this "
				   << "Setup";
    } else {
      HCalSD* theCaloSD = dynamic_cast<HCalSD*>(aSD);
      edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::beginOfRun: Finds SD "
				<< "with name " << theCaloSD->GetName() 
				<< " in this Setup";
      HcalNumberingScheme* org = new HcalTestNumberingScheme(false);
      theCaloSD->setNumberingScheme(org);
      edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::beginOfRun: set a "
				<< "new numbering scheme";
    }
    if (!hcalOnly) {
      sdname = names[1];
      aSD = sd->FindSensitiveDetector(sdname);
      if (aSD==0) {
	edm::LogWarning("HcalTBSim") << "HcalTB04Analysis::beginOfRun: No SD"
				     << " with name " << sdname << " in this "
				     << "Setup";
      } else {
	ECalSD* theCaloSD = dynamic_cast<ECalSD*>(aSD);
	edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::beginOfRun: Finds SD "
				  << "with name " << theCaloSD->GetName() 
				  << " in this Setup";
	EcalNumberingScheme* org = new HcalTB04XtalNumberingScheme();
	theCaloSD->setNumberingScheme(org);
	edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::beginOfRun: set a "
				  << "new numbering scheme";
      }
    }
  } else {
    edm::LogWarning("HcalTBSim") << "HcalTB04Analysis::beginOfRun: Could "
				 << "not get SD Manager!";
  }

}

void HcalTB04Analysis::update(const BeginOfEvent * evt) {
 
  evNum = (*evt) ()->GetEventID ();
  clear();
  edm::LogInfo("HcalTBSim") << "HcalTB04Analysis: =====> Begin of event = "
			    << evNum;
}

void HcalTB04Analysis::update(const G4Step * aStep) {

  if (aStep != NULL) {
    //Get Step properties
    G4ThreeVector thePreStepPoint  = aStep->GetPreStepPoint()->GetPosition();
    G4ThreeVector thePostStepPoint;

    // Get Tracks properties
    G4Track*      aTrack   = aStep->GetTrack();
    int           trackID  = aTrack->GetTrackID();
    int           parentID = aTrack->GetParentID();
    G4ThreeVector position = aTrack->GetPosition();
    G4ThreeVector momentum = aTrack->GetMomentum();
    G4String      partType = aTrack->GetDefinition()->GetParticleType();
    G4String      partSubType = aTrack->GetDefinition()->GetParticleSubType();
    int    partPDGEncoding = aTrack->GetDefinition()->GetPDGEncoding();
#ifdef ddebug
    bool   isPDGStable = aTrack->GetDefinition()->GetPDGStable();
#endif
    double pDGlifetime = aTrack->GetDefinition()->GetPDGLifeTime();
    double gammaFactor = aStep->GetPreStepPoint()->GetGamma();

    if (!pvFound) { //search for v1
      double stepDeltaEnergy = aStep->GetDeltaEnergy ();
      double kinEnergy = aTrack->GetKineticEnergy ();
      
      // look for DeltaE > 10% kinEnergy of particle, or particle death - Ek=0
      if (trackID == 1 && parentID == 0 && 
	  ((kinEnergy == 0.) || (fabs (stepDeltaEnergy / kinEnergy) > 0.1))) {
	pvType = -1;
	if (kinEnergy == 0.) {
	  pvType = 0;
	} else {
	  if (fabs (stepDeltaEnergy / kinEnergy) > 0.1) pvType = 1;
	}
	pvFound    = true;
	pvPosition = position;
	pvMomentum = momentum;
	// Rotated coord.system:
	pvUVW      = (*beamline_RM)*(pvPosition);

	//Volume name requires some checks:
	G4String thePostPVname = "NoName";
	G4StepPoint * thePostPoint = aStep->GetPostStepPoint ();
	if (thePostPoint) {
	  thePostStepPoint = thePostPoint->GetPosition();
	  G4VPhysicalVolume * thePostPV = thePostPoint->GetPhysicalVolume ();
	  if (thePostPV) thePostPVname = thePostPV->GetName ();
	}
#ifdef ddebug
	LogDebug("HcalTBSim") << "HcalTB04Analysis:: V1 found at: " 
			      << thePostStepPoint << " G4VPhysicalVolume: " 
			      << thePostPVname;
#endif      
	LogDebug("HcalTBSim") << "HcalTB04Analysis::fill_v1Pos: Primary Track "
			      << "momentum: " << pvMomentum << " psoition " 
			      << pvPosition << " u/v/w " << pvUVW;
      }
    } else { 
      // watch for secondaries originating @v1, including the surviving primary
      if ((trackID != 1 && parentID == 1 &&
	   (aTrack->GetCurrentStepNumber () == 1) && 
	   (thePreStepPoint == pvPosition)) || 
	  (trackID == 1 && thePreStepPoint == pvPosition)) {
#ifdef ddebug
	LogDebug("HcalTBSim") << "HcalTB04Analysis::A secondary...  PDG:" 
			      << partPDGEncoding << " TrackID:" << trackID
			      << " ParentID:" << parentID << " stable: "  
			      << isPDGStable << " Tau: " << pDGlifetime 
			      << " cTauGamma=" 
			      << c_light*pDGlifetime*gammaFactor*1000.
			      << "um" << " GammaFactor: " << gammaFactor;
#endif      
	secTrackID.push_back(trackID);
	secPartID.push_back(partPDGEncoding);
	secMomentum.push_back(momentum);
	secEkin.push_back(aTrack->GetKineticEnergy());

	// Check for short-lived secondaries: cTauGamma<100um
	double ctaugamma_um = c_light * pDGlifetime * gammaFactor * 1000.;
	if ((ctaugamma_um>0.) && (ctaugamma_um<100.)) {//short-lived secondary
	  shortLivedSecondaries.push_back(trackID);
      } else {//normal secondary - enter into the V1-calorimetric tree
	//          histos->fill_v1cSec (aTrack);
      }
      }
      // Also watch for tertiary particles coming from 
      // short-lived secondaries from V1
      if (aTrack->GetCurrentStepNumber() == 1) {
	if (shortLivedSecondaries.size() > 0) {
	  int pid = parentID;
	  std::vector<int>::iterator pos1= shortLivedSecondaries.begin();
	  std::vector<int>::iterator pos2 = shortLivedSecondaries.end();
	  std::vector<int>::iterator pos;
	  for (pos = pos1; pos != pos2; pos++) {
	    if (*pos == pid) {//ParentID is on the list of short-lived 
	      // secondary 
#ifdef ddebug
	      LogDebug("HcalTBSim") << "HcalTB04Analysis:: A tertiary...  PDG:"
				    << partPDGEncoding << " TrackID:" <<trackID
				    << " ParentID:" << parentID << " stable: "
				    << isPDGStable << " Tau: " << pDGlifetime
				    << " cTauGamma=" 
				    << c_light*pDGlifetime*gammaFactor*1000. 
				    << "um GammaFactor: " << gammaFactor;
#endif
	    }
	  }
	}
      }
    }
  }
}

void HcalTB04Analysis::update(const EndOfEvent * evt) {

  count++;

  //fill the buffer
  LogDebug("HcalTBSim") << "HcalTB04Analysis::Fill event " 
			<< (*evt)()->GetEventID();
  fillBuffer (evt);

  //QIE analysis
  LogDebug("HcalTBSim") << "HcalTB04Analysis::Do QIE analysis with " 
			<< hcalHitCache.size() << " hits";
  qieAnalysis();

  //Energy in Crystal Matrix
  if (!hcalOnly) {
    LogDebug("HcalTBSim") << "HcalTB04Analysis::Do Xtal analysis with " 
			  << ecalHitCache.size() << " hits";
    xtalAnalysis();
  }
  
  //Final Analysis
  LogDebug("HcalTBSim") << "HcalTB04Analysis::Final analysis";  
  finalAnalysis();

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10) 
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Event " << iEvt;
  else if ((iEvt < 100) && (iEvt%10 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt%100 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt%1000 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis:: Event " << iEvt;
}

void HcalTB04Analysis::fillBuffer(const EndOfEvent * evt) {

  std::vector<CaloHit> hhits, hhitl;
  int                  idHC, j;
  CaloG4HitCollection* theHC;
  std::map<int,float,std::less<int> > primaries;
  double               etot1=0, etot2=0;

  // Look for the Hit Collection of HCal
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  std::string sdName = names[0];
  idHC  = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (CaloG4HitCollection*) allHC->GetHC(idHC);
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Hit Collection for " << sdName
			<< " of ID " << idHC << " is obtained at " << theHC;

  if (idHC >= 0 && theHC > 0) {
    hhits.reserve(theHC->entries());
    hhitl.reserve(theHC->entries());
    for (j = 0; j < theHC->entries(); j++) {
      CaloG4Hit* aHit = (*theHC)[j]; 
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      math::XYZPoint pos  = aHit->getEntry();
      unsigned int id = aHit->getUnitID();
      double theta    = pos.theta();
      double eta      = -log(tan(theta * 0.5));
      double phi      = pos.phi();
      int det, z, group, ieta, iphi, layer;
      HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,layer);
      double jitter   = time-timeOfFlight(det,layer,eta);
      if (jitter<0) jitter = 0;
      if (e < 0 || e > 1.) e = 0;
      double escl     = e * scale(det,layer);
      unsigned int idx= HcalTBNumberingScheme::getUnitID(id,mode);
      CaloHit hit(det,layer,escl,eta,phi,jitter,idx);
      hhits.push_back(hit);
      CaloHit hitl(det,layer,escl,eta,phi,jitter,id);
      hhitl.push_back(hitl);
      primaries[aHit->getTrackID()]+= e;
      etot1 += escl;
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB04Analysis:: Hcal Hit i/p " << j 
			    << "  ID 0x" << std::hex << id << " 0x" << idx  
			    << std::dec << " time " << std::setw(6) << time 
			    << " "  << std::setw(6) << jitter << " theta " 
			    << std::setw(8) << theta << " eta " << std::setw(8)
			    << eta << " phi " << std::setw(8) << phi << " e " 
			    << std::setw(8) << e << " " << std::setw(8) <<escl;
#endif
    }
  }

  // Add hits in the same channel within same time slice
  std::vector<CaloHit>::iterator itr;
  int nHit = hhits.size();
  std::vector<CaloHit*> hits(nHit);
  for (j = 0, itr = hhits.begin(); itr != hhits.end(); j++, itr++) {
    hits[j] = &hhits[j];
  }
  sort(hits.begin(),hits.end(),CaloHitIdMore());
  std::vector<CaloHit*>::iterator k1, k2;
  int nhit = 0;
  for (k1 = hits.begin(); k1 != hits.end(); k1++) {
    int      det    = (**k1).det();
    int      layer  = (**k1).layer();
    double   ehit   = (**k1).e();
    double   eta    = (**k1).eta();
    double   phi    = (**k1).phi();
    double   jitter = (**k1).t();
    uint32_t unitID = (**k1).id();
    int      jump  = 0;
    for (k2 = k1+1; k2 != hits.end() && fabs(jitter-(**k2).t())<1 &&
           unitID==(**k2).id(); k2++) {
      ehit += (**k2).e();
      jump++;
    }
    nhit++;
    CaloHit hit(det, layer, ehit, eta, phi, jitter, unitID);
    hcalHitCache.push_back(hit);
    etot2 += ehit;
    k1    += jump;
#ifdef ddebug
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: Hcal Hit store " << nhit 
			  << "  ID 0x" << std::hex  << unitID  << std::dec 
			  << " time " << std::setw(6) << jitter << " eta "
			  << std::setw(8) << eta << " phi " << std::setw(8) 
			  << phi  << " e " << std::setw(8) << ehit;
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Stores " << nhit << " HCal hits"
			<< " from " << nHit << " input hits E(Hcal) " << etot1 
			<< " " << etot2;

  //Repeat for Hit in each layer (hhits and hhitl sizes are the same)
  for (j = 0, itr = hhitl.begin(); itr != hhitl.end(); j++, itr++) {
    hits[j] = &hhitl[j];
  }
  sort(hits.begin(),hits.end(),CaloHitIdMore());
  int    nhitl = 0;
  double etotl = 0;
  for (k1 = hits.begin(); k1 != hits.end(); k1++) {
    int      det    = (**k1).det();
    int      layer  = (**k1).layer();
    double   ehit   = (**k1).e();
    double   eta    = (**k1).eta();
    double   phi    = (**k1).phi();
    double   jitter = (**k1).t();
    uint32_t unitID = (**k1).id();
    int      jump  = 0;
    for (k2 = k1+1; k2 != hits.end() && fabs(jitter-(**k2).t())<1 &&
           unitID==(**k2).id(); k2++) {
      ehit += (**k2).e();
      jump++;
    }
    nhitl++;
    CaloHit hit(det, layer, ehit, eta, phi, jitter, unitID);
    hcalHitLayer.push_back(hit);
    etotl += ehit;
    k1    += jump;
#ifdef ddebug
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: Hcal Hit store " << nhitl 
			  << "  ID 0x" << std::hex << unitID  << std::dec 
			  << " time " << std::setw(6) << jitter << " eta "
			  << std::setw(8) << eta << " phi " << std::setw(8) 
			  << phi << " e " << std::setw(8) << ehit;
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Stores " << nhitl << " HCal "
			<< "hits from " << nHit << " input hits E(Hcal) " 
			<< etot1 << " " << etotl;
  
  // Look for the Hit Collection of ECal
  std::vector<CaloHit> ehits;
  sdName= names[1];
  idHC  = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (CaloG4HitCollection*) allHC->GetHC(idHC);
  etot1 = etot2 = 0;
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Hit Collection for " << sdName
			<< " of ID " << idHC << " is obtained at " << theHC;
  if (idHC >= 0 && theHC > 0) {
    ehits.reserve(theHC->entries());
    for (j = 0; j < theHC->entries(); j++) {
      CaloG4Hit* aHit = (*theHC)[j]; 
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      math::XYZPoint pos  = aHit->getEntry();
      unsigned int id = aHit->getUnitID();
      double theta    = pos.theta();
      double eta      = -log(tan(theta * 0.5));
      double phi      = pos.phi();
      if (e < 0 || e > 100000.) e = 0;
      int det, z, group, ieta, iphi, layer;
      HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,layer);
      CaloHit hit(det,0,e,eta,phi,time,id);
      ehits.push_back(hit);
      primaries[aHit->getTrackID()]+= e;
      etot1 += e;
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB04Analysis:: Ecal Hit i/p " << j 
			    << "  ID 0x" << std::hex << id  << std::dec 
			    << " time " << std::setw(6) << time << " theta " 
			    << std::setw(8) << theta  << " eta " <<std::setw(8)
			    << eta  << " phi " << std::setw(8) << phi << " e "
			    << std::setw(8) << e;
#endif
    }
  }

  // Add hits in the same channel within same time slice
  nHit = ehits.size();
  std::vector<CaloHit*> hite(nHit);
  for (j = 0, itr = ehits.begin(); itr != ehits.end(); j++, itr++) {
    hite[j] = &ehits[j];
  }
  sort(hite.begin(),hite.end(),CaloHitIdMore());
  nhit = 0;
  for (k1 = hite.begin(); k1 != hite.end(); k1++) {
    int      det    = (**k1).det();
    int      layer  = (**k1).layer();
    double   ehit   = (**k1).e();
    double   eta    = (**k1).eta();
    double   phi    = (**k1).phi();
    double   jitter = (**k1).t();
    uint32_t unitID = (**k1).id();
    int      jump  = 0;
    for (k2 = k1+1; k2 != hite.end() && fabs(jitter-(**k2).t())<1 &&
           unitID==(**k2).id(); k2++) {
      ehit += (**k2).e();
      jump++;
    }
    nhit++;
    CaloHit hit(det, layer, ehit, eta, phi, jitter, unitID);
    ecalHitCache.push_back(hit);
    etot2 += ehit;
    k1    += jump;
#ifdef ddebug
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: Ecal Hit store " << nhit
			  << "  ID 0x" << std::hex << unitID  << std::dec 
			  << " time " << std::setw(6) << jitter << " eta "
			  << std::setw(8) << eta << " phi " << std::setw(8)
			  << phi << " e " << std::setw(8) << ehit;
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Stores " << nhit << " ECal hits"
			<< " from " << nHit << " input hits E(Ecal) " << etot1 
			<< " " << etot2;

  // Find Primary info:
  nPrimary    = (int)(primaries.size());
  int trackID = 0;
  G4PrimaryParticle* thePrim=0;
  int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Event has " << nvertex 
			<< " verteices";
  if (nvertex<=0)
    edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::EndOfEvent ERROR: no "
			      << "vertex found for event " << evNum;

  for (int i = 0 ; i<nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == 0) {
      edm::LogInfo("HcalTBSim") << "HcalTB04Analysis::EndOfEvent ERR: pointer "
				<< "to vertex = 0 for event " << evNum;
    } else {
      LogDebug("HcalTBSim") << "HcalTB04Analysis::Vertex number :" << i << " "
			    << avertex->GetPosition();
      int npart = avertex->GetNumberOfParticle();
      if (npart == 0)
	edm::LogWarning("HcalTBSim") << "HcalTB04Analysis::End Of Event ERR: "
				     << "no primary!";
      if (thePrim==0) thePrim=avertex->GetPrimary(trackID);
    }
  }
    
  if (thePrim != 0) {
    double px = thePrim->GetPx();
    double py = thePrim->GetPy();
    double pz = thePrim->GetPz();
    double p  = std::sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
    pInit     = p/GeV;
    if (p==0) 
      edm::LogWarning("HcalTBSim") << "HcalTB04Analysis:: EndOfEvent ERR: "
				   << "primary has p=0 ";
    else {
      double costheta = pz/p;
      double theta = acos(std::min(std::max(costheta,-1.),1.));
      etaInit = -log(tan(theta/2));
      if (px != 0 || py != 0) phiInit = atan2(py,px);  
    }
    particleType = thePrim->GetPDGcode();
  } else 
    edm::LogWarning("HcalTBSim") << "HcalTB04Analysis::EndOfEvent ERR: could "
				 << "not find primary";

}

void HcalTB04Analysis::qieAnalysis() {

  int hittot = hcalHitCache.size();
  if (hittot<=0) hittot = 1;
  std::vector<CaloHit> hits(hittot);
  std::vector<int>     todo(nTower,0);

  LogDebug("HcalTBSim") << "HcalTB04Analysis::qieAnalysis: Size " 
			<< hits.size() << " " << todo.size() << " " 
			<< idTower.size() << " " << esimh.size() << " " 
			<< eqie.size();
  // Loop over all HCal hits
  for (unsigned int k1 = 0; k1 < hcalHitCache.size(); k1++) {
    CaloHit hit = hcalHitCache[k1];
    uint32_t id = hit.id();
    int    nhit = 0;
    double esim = hit.e();
    hits[nhit]  = hit;
    for (unsigned int k2 = k1+1; k2 < hcalHitCache.size(); k2++) {
      hit = hcalHitCache[k2];
      if (hit.id() == id) {
	nhit++;
	hits[nhit] = hit;
	esim += hit.e();
      }
    }
    k1 += nhit;
    nhit++;
    std::vector<int> cd = myQie->getCode(nhit,hits);
    double eq = myQie->getEnergy(cd);
    LogDebug("HcalTBSim") << "HcalTB04Analysis::  ID 0x" << std::hex << id 
			  << std::dec << " registers " << esim << " energy "
			  << "from " << nhit << " hits starting with hit # " 
			  << k1 << " energy with noise " << eq;
    for (int k2 = 0; k2 < nTower; k2++) {
      if (id == idTower[k2]) {
	todo[k2]  = 1;
	esimh[k2] = esim;
	eqie[k2]  = eq;
      }
    }
  }
  
  // Towers with no hit
  for (int k2 = 0; k2 < nTower; k2++) {
    if (todo[k2] == 0) {
      std::vector<int> cd = myQie->getCode(0,hits);
      double eq = myQie->getEnergy(cd);
      esimh[k2] = 0;
      eqie[k2]  = eq;
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB04Analysis::  ID 0x" << std::hex 
			    << idTower[k2] << std::dec << " registers " 
			    << esimh[k2] << " energy from hits and energy "
			    << "after QIE analysis " << eqie[k2];
#endif
    }
  }
}

void HcalTB04Analysis::xtalAnalysis() {

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "HcalTB04Analysis requires the RandomNumberGeneratorService\n"
      << "which is not present in the configuration file. "
      << "You must add the service\n in the configuration file or "
      << "remove the modules that require it.";
  }
  CLHEP::RandGaussQ  randGauss(rng->getEngine());

  // Crystal Data
  std::vector<int> iok(nCrystal,0);
  LogDebug("HcalTBSim") << "HcalTB04Analysis::xtalAnalysis: Size " <<iok.size()
			<< " " << idEcal.size() << " " << esime.size() << " " 
			<< enois.size();
  for (unsigned int k1 = 0; k1 < ecalHitCache.size(); k1++) {
    uint32_t id   = ecalHitCache[k1].id();
    int      nhit = 0;
    double   esim = ecalHitCache[k1].e();
    for (unsigned int k2 = k1+1; k2 < ecalHitCache.size(); k2++) {
      if (ecalHitCache[k2].id() == id) {
	nhit++;
	esim += ecalHitCache[k2].e();
      }
    }
    k1 += nhit;
    nhit++;
    double eq = esim + randGauss.fire(0., ecalNoise);
#ifdef ddebug
    LogDebug("HcalTBSim") << "HcalTB04Analysis::  ID 0x" << std::hex << id 
			  << std::dec << " registers " << esim << " energy "
			  << "from " << nhit << " hits starting with hit # "
			  << k1 << " energy with noise " << eq;
#endif
    for (int k2 = 0; k2 < nCrystal; k2++) {
      if (id == idEcal[k2]) {
	iok[k2]   = 1;
	esime[k2] = esim;
	enois[k2] = eq;
      }
    }
  }
    
  // Crystals with no hit
  for (int k2 = 0; k2 < nCrystal; k2++) {
    if (iok[k2] == 0) {
      esime[k2] = 0;
      enois[k2] = randGauss.fire(0., ecalNoise);
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB04Analysis::  ID 0x" << std::hex 
			    << idEcal[k2] << std::dec << " registers " 
			    << esime[k2] << " energy from hits and energy from"
			    << " noise " << enois[k2];
#endif
    }
  }
}

void HcalTB04Analysis::finalAnalysis() {

  //Beam Information
  histo->fillPrimary(pInit, etaInit, phiInit);

  // Total Energy
  eecals = ehcals = eecalq = ehcalq = 0.;
  for (int i=0; i<nTower; i++) {
    ehcals += esimh[i];
    ehcalq += eqie[i];
  }
  for (int i=0; i<nCrystal; i++) {
    eecals += esime[i];
    eecalq += enois[i];
  }
  etots = eecals + ehcals;
  etotq = eecalq + ehcalq;
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Energy deposit at Sim Level "
			<< "(Total) " << etots << " (ECal) " << eecals 
			<< " (HCal) " << ehcals << "\nHcalTB04Analysis:: "
			<< "Energy deposit at Qie Level (Total) " << etotq
			<< " (ECal) " << eecalq << " (HCal) " << ehcalq;
  histo->fillEdep(etots, eecals, ehcals, etotq, eecalq, ehcalq);

  // Lateral Profile
  for (int i=0; i<5; i++) {
    eseta[i] = 0.;
    eqeta[i] = 0.;
  }
  for (int i=0; i<3; i++) {
    esphi[i] = 0.;
    eqphi[i] = 0.;
  }
  double e1=0, e2=0;
  unsigned int id;
  for (int i=0; i<nTower; i++) {
    int det, z, group, ieta, iphi, layer;
    id = idTower[i];
    HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,layer);
    iphi -= (icphi - 1);
    if (icphi > 4) {
      if (ieta == 0) ieta = 2;
      else           ieta =-1;
    } else {
      ieta = ieta - iceta + 2;
    }
    if (iphi >= 0 && iphi < 3 && ieta >= 0 && ieta < 5) {
      eseta[ieta] += esimh[i];
      esphi[iphi] += esimh[i];
      e1          += esimh[i];
      eqeta[ieta] += eqie[i];
      eqphi[iphi] += eqie[i];
      e2          += eqie[i];
    }
  }
  for (int i=0; i<3; i++) {
    if (e1>0) esphi[i] /= e1;
    if (e2>0) eqphi[i] /= e2;
  }
  for (int i=0; i<5; i++) {
    if (e1>0) eseta[i] /= e1;
    if (e2>0) eqeta[i] /= e2;
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Energy fraction along Eta and" 
			<< " Phi (Sim/Qie)";
  for (int i=0; i<5; i++) 
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: [" << i << "] Eta Sim = "
			  << eseta[i] << " Qie = " << eqeta[i] << " Phi Sim = "
			  << esphi[i] << " Qie = " << eqphi[i];
  histo->fillTrnsProf(eseta,eqeta,esphi,eqphi);

  // Longitudianl profile
  for (int i=0; i<20; i++) {
    eslay[i] = 0.;
    eqlay[i] = 0.;
  }
  e1=0; e2=0;
  for (int i=0; i<nTower; i++) {
    int det, z, group, ieta, iphi, layer;
    id = idTower[i];
    HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,layer);
    iphi  -= (icphi - 1);
    layer -= 1;
    if (iphi >= 0 && iphi < 3 && layer >= 0 && layer < 20) {
      eslay[layer] += esimh[i];
      e1           += esimh[i];
      eqlay[layer] += eqie[i];
      e2           += eqie[i];
    }
  }
  for (int i=0; i<20; i++) {
    if (e1>0) eslay[i] /= e1;
    if (e2>0) eqlay[i] /= e2;
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Energy fraction along Layer";
  for (int i=0; i<20; i++)
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: [" << i << "] Sim = " 
			  << eslay[i] << " Qie = " << eqlay[i];
  histo->fillLongProf(eslay, eqlay);
}


void HcalTB04Analysis::fillEvent (PHcalTB04Info& product) {

  //Setup the ID's
  product.setIDs(idHcal, idXtal);

  //Beam Information
  product.setPrimary(nPrimary, particleType, pInit, etaInit, phiInit);

  //Energy deposits in the crystals and towers
  product.setEdepHcal(esimh, eqie);
  product.setEdepHcal(esime, enois);

  // Total Energy
  product.setEdep(etots, eecals, ehcals, etotq, eecalq, ehcalq);

  // Lateral Profile
  product.setTrnsProf(eseta,eqeta,esphi,eqphi);

  // Longitudianl profile
  product.setLongProf(eslay, eqlay);

  //Save Hits
  int i, nhit=0;
  std::vector<CaloHit>::iterator itr;
  for (i=0, itr=ecalHitCache.begin(); itr!=ecalHitCache.end(); i++,itr++) {
    uint32_t id = itr->id();
    int det, z, group, ieta, iphi, lay;
    HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,lay);
    product.saveHit(det, lay, ieta, iphi, itr->e(), itr->t());
    nhit++;
#ifdef debug
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: Save Hit " << std::setw(3) 
			  << i+1 << " ID 0x" << std::hex << group << std::dec 
			  << " "  << std::setw(2) << det << " " << std::setw(2)
			  << lay  << " " << std::setw(1) << z << " " 
			  << std::setw(3) << ieta << " " << std::setw(3) <<iphi
			  << " T " << std::setw(6) << itr->t() << " E " 
			  << std::setw(6) << itr->e();
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Saves " << nhit 
			<< " hits from Crystals";
  int hit = nhit;
  nhit = 0;

  for (i=hit, itr=hcalHitCache.begin(); itr!=hcalHitCache.end(); i++,itr++) {
    uint32_t id = itr->id();
    int det, z, group, ieta, iphi, lay;
    HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,lay);
    product.saveHit(det, lay, ieta, iphi, itr->e(), itr->t());
    nhit++;
#ifdef debug
    LogDebug("HcalTBSim") << "HcalTB04Analysis:: Save Hit " << std::setw(3) 
			  << i+1 << " ID 0x" << std::hex << group << std::dec 
			  << " "  << std::setw(2) << det << " " << std::setw(2)
			  << lay  << " " << std::setw(1) << z << " " 
			  << std::setw(3) << ieta << " " << std::setw(3) <<iphi
			  << " T " << std::setw(6) << itr->t() << " E " 
			  << std::setw(6) << itr->e();
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB04Analysis:: Saves " << nhit 
			<< " hits from HCal";

  //Vertex associated quantities
  product.setVtxPrim(evNum, pvType, pvPosition.x(), pvPosition.y(), 
		     pvPosition.z(), pvUVW.x(), pvUVW.y(), pvUVW.z(),
		     pvMomentum.x(), pvMomentum.y(), pvMomentum.z());
  for (unsigned int i = 0; i < secTrackID.size(); i++) {
    product.setVtxSec(secTrackID[i], secPartID[i], secMomentum[i].x(),
		      secMomentum[i].y(), secMomentum[i].z(), secEkin[i]);
  }
}

void HcalTB04Analysis::clear(){
  pvFound = false;
  pvType  =-2;
  pvPosition = G4ThreeVector();
  pvMomentum = G4ThreeVector();
  pvUVW      = G4ThreeVector();
  secTrackID.clear();
  secPartID.clear();
  secMomentum.clear();
  secEkin.clear();
  shortLivedSecondaries.clear();

  ecalHitCache.erase(ecalHitCache.begin(), ecalHitCache.end()); 
  hcalHitCache.erase(hcalHitCache.begin(), hcalHitCache.end()); 
  hcalHitLayer.erase(hcalHitLayer.begin(), hcalHitLayer.end());
  nPrimary = particleType = 0;
  pInit = etaInit = phiInit = 0;

  esimh.clear();
  eqie.clear();
  esimh.reserve(nTower);
  eqie.reserve(nTower);
  for (int i=0; i<nTower; i++) {
    esimh.push_back(0.);
    eqie.push_back(0.);
  }
  esime.clear();
  enois.clear();
  esime.reserve(nCrystal);
  enois.reserve(nCrystal);
  for (int i=0; i<nCrystal; i++) {
    esime.push_back(0.);
    enois.push_back(0.);
  }
}

int HcalTB04Analysis::unitID(uint32_t id) {

  int det, z, group, ieta, iphi, lay;
  HcalTestNumbering::unpackHcalIndex(id,det,z,group,ieta,iphi,lay);
  group  = (det&15)<<20;
  group += ((lay-1)&31)<<15;
  group += (z&1)<<14;
  group += (ieta&127)<<7;
  group += (iphi&127);
  return group;
}

double HcalTB04Analysis::scale(int det, int layer) {

  double tmp = 1.;
  if (det == static_cast<int>(HcalBarrel)) {
    if (layer == 1)       tmp = scaleHB0;
    else if (layer == 17) tmp = scaleHB16;
    else if (layer > 17)  tmp = scaleHO;
  } else {
    if (layer <= 2)       tmp = scaleHE0;
  }
  return tmp;
}

double HcalTB04Analysis::timeOfFlight(int det, int layer, double eta) {

  double theta = 2.0*atan(exp(-eta));
  double dist  = beamOffset;
  if (det == static_cast<int>(HcalBarrel)) {
    const double rLay[19] = {
      1836.0, 1902.0, 1962.0, 2022.0, 2082.0, 2142.0, 2202.0, 2262.0, 2322.0, 
      2382.0, 2448.0, 2514.0, 2580.0, 2646.0, 2712.0, 2776.0, 2862.5, 3847.0,
      4052.0};
    if (layer>0 && layer<=19) dist += rLay[layer-1]*mm/sin(theta);
  } else {
    const double zLay[19] = {
      4034.0, 4032.0, 4123.0, 4210.0, 4297.0, 4384.0, 4471.0, 4558.0, 4645.0, 
      4732.0, 4819.0, 4906.0, 4993.0, 5080.0, 5167.0, 5254.0, 5341.0, 5428.0,
      5515.0};
    if (layer>0 && layer<=19) dist += zLay[layer-1]*mm/cos(theta);
  }

  double tmp = dist/c_light/ns;
#ifdef ddebug
  LogDebug("HcalTBSim") << "HcalTB04Analysis::timeOfFlight " << tmp 
			<< " for det/lay " << det  << " " << layer 
			<< " eta/theta " << eta << " " << theta/deg 
			<< " dist " << dist;
#endif
  return tmp;
}
 
DEFINE_SIMWATCHER (HcalTB04Analysis);
