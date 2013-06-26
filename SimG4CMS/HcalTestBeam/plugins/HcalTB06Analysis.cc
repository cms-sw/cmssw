// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Analysis
//
// Implementation:
//     Main analysis class for Hcal Test Beam 2006 Analysis
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: HcalTB06Analysis.cc,v 1.1 2012/07/02 04:44:40 sunanda Exp $
//
  
// system include files
#include <cmath>
#include <iostream>
#include <iomanip>

// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Analysis.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

// to retreive hits
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//
// constructors and destructor
//

HcalTB06Analysis::HcalTB06Analysis(const edm::ParameterSet &p): histo(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("HcalTB06Analysis");
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

  produces<PHcalTB06Info>();

  beamline_RM = new G4RotationMatrix;
  beamline_RM->rotateZ(-beamPhi);
  beamline_RM->rotateY(-beamThet);
 
  edm::LogInfo("HcalTBSim") << "HcalTB06:: Initialised as observer of BeginOf"
			    << "Job/BeginOfRun/BeginOfEvent/G4Step/EndOfEvent"
			    << " with Parameter values:\n \tbeamOffset = " 
			    << beamOffset << "\ticeta = " << iceta 
			    << "\ticphi = " << icphi << "\n\tbeamline_RM = "
			    << *beamline_RM;

  init();

  histo  = new HcalTB06Histo(m_Anal);
} 
   
HcalTB06Analysis::~HcalTB06Analysis() {

  edm::LogInfo("HcalTBSim") << "\n -------->  Total number of selected entries"
			    << " : " << count << "\nPointers:: Histo " <<histo;
  if (histo)   {
    delete histo;
    histo  = 0;
  }
}

//
// member functions
//

void HcalTB06Analysis::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<PHcalTB06Info> product(new PHcalTB06Info);
  fillEvent(*product);
  e.put(product);
}

void HcalTB06Analysis::init() {

  // counter 
  count = 0;
  evNum = 0;
  clear();
}

void HcalTB06Analysis::update(const BeginOfRun * run) {

  int irun = (*run)()->GetRunID();
  edm::LogInfo("HcalTBSim") <<" =====> Begin of Run = " << irun;
 
}

void HcalTB06Analysis::update(const BeginOfEvent * evt) {
 
  evNum = (*evt) ()->GetEventID ();
  clear();
  edm::LogInfo("HcalTBSim") << "HcalTB06Analysis: =====> Begin of event = "
			    << evNum;
}

void HcalTB06Analysis::update(const G4Step * aStep) {

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
	LogDebug("HcalTBSim") << "HcalTB06Analysis:: V1 found at: " 
			      << thePostStepPoint << " G4VPhysicalVolume: " 
			      << thePostPVname;
#endif      
	LogDebug("HcalTBSim") << "HcalTB06Analysis::fill_v1Pos: Primary Track "
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
	LogDebug("HcalTBSim") << "HcalTB06Analysis::A secondary...  PDG:" 
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
	      LogDebug("HcalTBSim") << "HcalTB06Analysis:: A tertiary...  PDG:"
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

void HcalTB06Analysis::update(const EndOfEvent * evt) {

  count++;

  //fill the buffer
  LogDebug("HcalTBSim") << "HcalTB06Analysis::Fill event " 
			<< (*evt)()->GetEventID();
  fillBuffer (evt);
  
  //Final Analysis
  LogDebug("HcalTBSim") << "HcalTB06Analysis::Final analysis";  
  finalAnalysis();

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10) 
    edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Event " << iEvt;
  else if ((iEvt < 100) && (iEvt%10 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt%100 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt%1000 == 0)) 
    edm::LogInfo("HcalTBSim") << "HcalTB06Analysis:: Event " << iEvt;
}

void HcalTB06Analysis::fillBuffer(const EndOfEvent * evt) {

  std::vector<CaloHit> hhits;
  int                  idHC, j;
  CaloG4HitCollection* theHC;
  std::map<int,float,std::less<int> > primaries;
  double               etot1=0, etot2=0;

  // Look for the Hit Collection of HCal
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  std::string sdName = names[0];
  idHC  = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (CaloG4HitCollection*) allHC->GetHC(idHC);
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Hit Collection for " << sdName
			<< " of ID " << idHC << " is obtained at " << theHC;

  if (idHC >= 0 && theHC > 0) {
    hhits.reserve(theHC->entries());
    for (j = 0; j < theHC->entries(); j++) {
      CaloG4Hit* aHit = (*theHC)[j]; 
      double e        = aHit->getEnergyDeposit()/GeV;
      double time     = aHit->getTimeSlice();
      math::XYZPoint pos  = aHit->getEntry();
      unsigned int id = aHit->getUnitID();
      double theta    = pos.theta();
      double eta      = -log(tan(theta * 0.5));
      double phi      = pos.phi();
      CaloHit hit(2,1,e,eta,phi,time,id);
      hhits.push_back(hit);
      primaries[aHit->getTrackID()]+= e;
      etot1 += e;
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB06Analysis:: Hcal Hit i/p " << j 
			    << "  ID 0x" << std::hex << id << std::dec 
			    << " time " << std::setw(6) << time << " theta "
			    << std::setw(8) << theta << " eta " << std::setw(8)
			    << eta << " phi " << std::setw(8) << phi << " e " 
			    << std::setw(8) << e;
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
    LogDebug("HcalTBSim") << "HcalTB06Analysis:: Hcal Hit store " << nhit 
			  << "  ID 0x" << std::hex  << unitID  << std::dec 
			  << " time " << std::setw(6) << jitter << " eta "
			  << std::setw(8) << eta << " phi " << std::setw(8) 
			  << phi  << " e " << std::setw(8) << ehit;
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Stores " << nhit << " HCal hits"
			<< " from " << nHit << " input hits E(Hcal) " << etot1 
			<< " " << etot2;
  
  // Look for the Hit Collection of ECal
  std::vector<CaloHit> ehits;
  sdName= names[1];
  idHC  = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (CaloG4HitCollection*) allHC->GetHC(idHC);
  etot1 = etot2 = 0;
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Hit Collection for " << sdName
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
      CaloHit hit(1,0,e,eta,phi,time,id);
      ehits.push_back(hit);
      primaries[aHit->getTrackID()]+= e;
      etot1 += e;
#ifdef ddebug
      LogDebug("HcalTBSim") << "HcalTB06Analysis:: Ecal Hit i/p " << j 
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
    LogDebug("HcalTBSim") << "HcalTB06Analysis:: Ecal Hit store " << nhit
			  << "  ID 0x" << std::hex << unitID  << std::dec 
			  << " time " << std::setw(6) << jitter << " eta "
			  << std::setw(8) << eta << " phi " << std::setw(8)
			  << phi << " e " << std::setw(8) << ehit;
#endif
  }
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Stores " << nhit << " ECal hits"
			<< " from " << nHit << " input hits E(Ecal) " << etot1 
			<< " " << etot2;

  // Find Primary info:
  nPrimary    = (int)(primaries.size());
  int trackID = 0;
  G4PrimaryParticle* thePrim=0;
  int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Event has " << nvertex 
			<< " verteices";
  if (nvertex<=0)
    edm::LogInfo("HcalTBSim") << "HcalTB06Analysis::EndOfEvent ERROR: no "
			      << "vertex found for event " << evNum;

  for (int i = 0 ; i<nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == 0) {
      edm::LogInfo("HcalTBSim") << "HcalTB06Analysis::EndOfEvent ERR: pointer "
				<< "to vertex = 0 for event " << evNum;
    } else {
      LogDebug("HcalTBSim") << "HcalTB06Analysis::Vertex number :" << i << " "
			    << avertex->GetPosition();
      int npart = avertex->GetNumberOfParticle();
      if (npart == 0)
	edm::LogWarning("HcalTBSim") << "HcalTB06Analysis::End Of Event ERR: "
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
      edm::LogWarning("HcalTBSim") << "HcalTB06Analysis:: EndOfEvent ERR: "
				   << "primary has p=0 ";
    else {
      double costheta = pz/p;
      double theta = acos(std::min(std::max(costheta,-1.),1.));
      etaInit = -log(tan(theta/2));
      if (px != 0 || py != 0) phiInit = atan2(py,px);  
    }
    particleType = thePrim->GetPDGcode();
  } else 
    edm::LogWarning("HcalTBSim") << "HcalTB06Analysis::EndOfEvent ERR: could "
				 << "not find primary";

}

void HcalTB06Analysis::finalAnalysis() {

  //Beam Information
  histo->fillPrimary(pInit, etaInit, phiInit);

  // Total Energy
  eecals = ehcals = 0.;
  for (unsigned int i=0; i<hcalHitCache.size(); i++) {
    ehcals += hcalHitCache[i].e();
  }
  for (unsigned int i=0; i<ecalHitCache.size(); i++) {
    eecals += ecalHitCache[i].e();
  }
  etots = eecals + ehcals;
  LogDebug("HcalTBSim") << "HcalTB06Analysis:: Energy deposit at Sim Level "
			<< "(Total) " << etots << " (ECal) " << eecals 
			<< " (HCal) " << ehcals;
  histo->fillEdep(etots, eecals, ehcals);
}


void HcalTB06Analysis::fillEvent (PHcalTB06Info& product) {

  //Beam Information
  product.setPrimary(nPrimary, particleType, pInit, etaInit, phiInit);

  // Total Energy
  product.setEdep(etots, eecals, ehcals);

  //Energy deposits in the crystals and towers
  for (unsigned int i=0; i<hcalHitCache.size(); i++) 
    product.saveHit(hcalHitCache[i].id(), hcalHitCache[i].eta(),
		    hcalHitCache[i].phi(), hcalHitCache[i].e(),
		    hcalHitCache[i].t());
  for (unsigned int i=0; i<ecalHitCache.size(); i++) 
    product.saveHit(ecalHitCache[i].id(), ecalHitCache[i].eta(),
		    ecalHitCache[i].phi(), ecalHitCache[i].e(),
		    ecalHitCache[i].t());

  //Vertex associated quantities
  product.setVtxPrim(evNum, pvType, pvPosition.x(), pvPosition.y(), 
		     pvPosition.z(), pvUVW.x(), pvUVW.y(), pvUVW.z(),
		     pvMomentum.x(), pvMomentum.y(), pvMomentum.z());
  for (unsigned int i = 0; i < secTrackID.size(); i++) {
    product.setVtxSec(secTrackID[i], secPartID[i], secMomentum[i].x(),
		      secMomentum[i].y(), secMomentum[i].z(), secEkin[i]);
  }
}

void HcalTB06Analysis::clear() {

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
  nPrimary = particleType = 0;
  pInit = etaInit = phiInit = 0;
}
 
DEFINE_SIMWATCHER (HcalTB06Analysis);
