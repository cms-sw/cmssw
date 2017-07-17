// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorTestAnalysis
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: P. Katsas
//         Created: 02/2007 
//

#include "SimG4CMS/Forward/interface/CastorTestAnalysis.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include <cmath>
#include <iostream>
#include <iomanip>

#define debugLog

enum ntcastors_elements {
  ntcastors_evt, ntcastors_trackid, ntcastors_charge, ntcastors_pdgcode, ntcastors_x, ntcastors_y, ntcastors_z, ntcastors_stepl, ntcastors_stepe, ntcastors_eta, ntcastors_phi, ntcastors_vpx, ntcastors_vpy, ntcastors_vpz
};

enum ntcastore_elements {
  ntcastore_evt, ntcastore_ihit, ntcastore_detector, ntcastore_sector, ntcastore_module, ntcastore_enem, ntcastore_enhad, ntcastore_hitenergy, ntcastore_x, ntcastore_y, ntcastore_z
};

CastorTestAnalysis::CastorTestAnalysis(const edm::ParameterSet &p) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("CastorTestAnalysis");
  verbosity                = m_Anal.getParameter<int>("Verbosity");
  doNTcastorstep           = m_Anal.getParameter<int>("StepNtupleFlag");
  doNTcastorevent          = m_Anal.getParameter<int>("EventNtupleFlag");
  stepNtFileName           = m_Anal.getParameter<std::string>("StepNtupleFileName");
  eventNtFileName          = m_Anal.getParameter<std::string>("EventNtupleFileName");

  if (verbosity > 0) {
      std::cout<<std::endl;
      std::cout<<"============================================================================"<<std::endl;
      std::cout << "CastorTestAnalysis:: Initialized as observer"<< std::endl;
      if (doNTcastorstep  > 0){
        std::cout <<" Step Ntuple will be created"<< std::endl;
        std::cout <<" Step Ntuple file: "<<stepNtFileName<<std::endl;
      }
      if (doNTcastorevent > 0){
        std::cout <<" Event Ntuple will be created"<< std::endl;
        std::cout <<" Step Ntuple file: "<<stepNtFileName<<std::endl;
      }
      std::cout<<"============================================================================"<<std::endl;
      std::cout<<std::endl;
  }
  if (doNTcastorstep  > 0)  
  castorstepntuple = new TNtuple("NTcastorstep","NTcastorstep","evt:trackid:charge:pdgcode:x:y:z:stepl:stepe:eta:phi:vpx:vpy:vpz");
  
  if (doNTcastorevent  > 0)
  castoreventntuple = new TNtuple("NTcastorevent","NTcastorevent","evt:ihit:detector:sector:module:enem:totalenergy:hitenergy:x:y:z");
}

CastorTestAnalysis::~CastorTestAnalysis() {
  //destructor of CastorTestAnalysis
    
    Finish();
  if (verbosity > 0) {
    std::cout << std::endl << "End of CastorTestAnalysis"
	      << std::endl; 
  }
  
  std::cout<<"CastorTestAnalysis: End of process"<<std::endl;
  
}
  
//=================================================================== per EVENT
void CastorTestAnalysis::update(const BeginOfJob * job) {

  std::cout << " Starting new job " << std::endl;
}

//==================================================================== per RUN
void CastorTestAnalysis::update(const BeginOfRun * run) {

 std::cout << std::endl << "CastorTestAnalysis: Starting Run"<< std::endl; 
  if (doNTcastorstep) { 
    std::cout << "CastorTestAnalysis: output step root file created"<< std::endl;
    TString stepfilename = stepNtFileName;
    castorOutputStepFile = new TFile(stepfilename,"RECREATE");

  }
  
  if (doNTcastorevent) {
    std::cout << "CastorTestAnalysis: output event root file created"<< std::endl;
    TString stepfilename = eventNtFileName;
    castorOutputEventFile = new TFile(stepfilename,"RECREATE");
  }

  eventIndex = 0;
}

void CastorTestAnalysis::update(const BeginOfEvent * evt) {
  std::cout << "CastorTestAnalysis: Processing Event Number: "<<eventIndex<< std::endl;
  eventIndex++;
  stepIndex = 0;
}




void CastorTestAnalysis::update(const G4Step * aStep) {
  stepIndex++;
  
  
  if (doNTcastorstep) {
  
  G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
//  G4StepPoint * postStepPoint= aStep->GetPostStepPoint();
  G4double stepL = aStep->GetStepLength();
  G4double stepE = aStep->GetTotalEnergyDeposit();
  
  if (verbosity >= 2) 
    std::cout << "Step " << stepL << ", " << stepE << std::endl;
    
  G4Track * theTrack    = aStep->GetTrack();
  G4int theTrackID      = theTrack->GetTrackID();
  G4double theCharge    = theTrack->GetDynamicParticle()->GetCharge();
  //  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  G4int pdgcode		= theTrack->GetDefinition()->GetPDGEncoding();

  G4ThreeVector vert_mom = theTrack->GetVertexMomentumDirection();
  G4double vpx = vert_mom.x();
  G4double vpy = vert_mom.y();
  G4double vpz = vert_mom.z();
  double eta = 0.5 * log( (1.+vpz) / (1.-vpz) );
  double phi = atan2(vpy,vpx);

  G4ThreeVector hitPoint = preStepPoint->GetPosition();

   // Fill-in ntuple
  //  castorsteparray[ntcastors_evt] = (*evt)()->GetEventID();
  castorsteparray[ntcastors_evt] = (float)eventIndex;
  castorsteparray[ntcastors_trackid] = (float)theTrackID;
  castorsteparray[ntcastors_charge] = theCharge;
  castorsteparray[ntcastors_pdgcode] = pdgcode;
  castorsteparray[ntcastors_x] = hitPoint.x();
  castorsteparray[ntcastors_y] = hitPoint.y();
  castorsteparray[ntcastors_z] = hitPoint.z();
  castorsteparray[ntcastors_stepl] = stepL;
  castorsteparray[ntcastors_stepe] = stepE;
  castorsteparray[ntcastors_eta] = eta;
  castorsteparray[ntcastors_phi] = phi;
  castorsteparray[ntcastors_vpx] = vpx;
  castorsteparray[ntcastors_vpy] = vpy;
  castorsteparray[ntcastors_vpz] = vpz;

  /*
  std::cout<<"TrackID: "<< theTrackID<<std::endl;
  std::cout<<"   StepN: "<< theTrack->GetCurrentStepNumber() <<std::endl;
  std::cout<<"      ParentID: "<< aStep->GetTrack()->GetParentID() <<std::endl;
  std::cout<<"      PDG: "<< pdgcode <<std::endl;
  std::cout<<"      X,Y,Z (mm): "<< theTrack->GetPosition().x() <<","<< theTrack->GetPosition().y() <<","<< theTrack->GetPosition().z() <<std::endl;
  std::cout<<"      KE (MeV): "<< theTrack->GetKineticEnergy() <<std::endl;
  std::cout<<"      Total EDep (MeV): "<< aStep->GetTotalEnergyDeposit() <<std::endl;
  std::cout<<"      StepLength (mm): "<< aStep->GetStepLength() <<std::endl;
  std::cout<<"      TrackLength (mm): "<< theTrack->GetTrackLength() <<std::endl;

  if ( theTrack->GetNextVolume() != 0 )
      std::cout<<"      NextVolume: "<< theTrack->GetNextVolume()->GetName() <<std::endl;
  else 
      std::cout<<"      NextVolume: OutOfWorld"<<std::endl;
  
  if(aStep->GetPostStepPoint()->GetProcessDefinedStep() != NULL)
      std::cout<<"      ProcessName: "<< aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName() <<std::endl;
  else
      std::cout<<"      ProcessName: UserLimit"<<std::endl;
  

   std::cout<<std::endl;
  */

#ifdef DebugLog
  if ( theTrack->GetNextVolume() != 0 )
    LogDebug("ForwardSim") << " NextVolume: " << theTrack->GetNextVolume()->GetName() ;
  else 
    LogDebug("ForwardSim") << " NextVolume: OutOfWorld" ;
#endif

 
//fill ntuple with step level information
  castorstepntuple->Fill(castorsteparray);
  }
}

//================= End of EVENT ===============
void CastorTestAnalysis::update(const EndOfEvent * evt) {

  // Look for the Hit Collection 
  std::cout << std::endl << "CastorTest::update(EndOfEvent * evt) - event #" << (*evt)()->GetEventID() << std::endl;

  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  std::cout << "update(*evt) --> accessed all HC";
  
  int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");
    
  CaloG4HitCollection* theCAFI = (CaloG4HitCollection*) allHC->GetHC(CAFIid);

  theCastorNumScheme = new CastorNumberingScheme();
  // CastorNumberingScheme *theCastorNumScheme = new CastorNumberingScheme();

/*
  unsigned int volumeID=0;
  int det, zside, sector, zmodule;
  std::map<int,float,std::less<int> > themap;
  double totalEnergy = 0;
  double hitEnergy = 0;
  double en_in_fi = 0.;
  double en_in_pl = 0.;
*/
//  double en_in_bu = 0.;
//  double en_in_tu = 0.;

  if (doNTcastorevent) {
    
    eventGlobalHit = 0 ;
    // int eventGlobalHit = 0 ;
    
    //  Check FI TBranch for Hits
    if (theCAFI->entries() > 0) getCastorBranchData(theCAFI) ;
    
    // Find Primary info:
      int trackID = 0;
      int particleType = 0;
      G4PrimaryParticle* thePrim=0;
      G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
      std::cout << "Event has " << nvertex << " vertex" << std::endl; 
      if (nvertex==0) std::cout << "CASTORTest End Of Event  ERROR: no vertex" << std::endl;

      for (int i = 0 ; i<nvertex; i++) {
        G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
        if (avertex == 0) 
	  std::cout << "CASTORTest End Of Event ERR: pointer to vertex = 0" << std::endl;
        std::cout << "Vertex number :" <<i << std::endl;
        int npart = avertex->GetNumberOfParticle();
        if (npart ==0)
	  std::cout << "CASTORTest End Of Event ERR: no primary!" << std::endl;
        if (thePrim==0) thePrim=avertex->GetPrimary(trackID);
      }
    
      double px=0.,py=0.,pz=0., pInit=0;
      double eta = 0., phi = 0.;
    
      if (thePrim != 0) {
        px = thePrim->GetPx();
        py = thePrim->GetPy();
        pz = thePrim->GetPz();
        pInit = sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
        if (pInit==0) {
	  std::cout << "CASTORTest End Of Event  ERR: primary has p=0 " << std::endl;
        } else {   
	  float costheta = pz/pInit;
	  float theta = acos(std::min(std::max(costheta,float(-1.)),float(1.)));
	  eta = -log(tan(theta/2));

	  if (px != 0) phi = atan(py/px);  
        }
	particleType	= thePrim->GetPDGcode();
      } else {
        std::cout << "CASTORTest End Of Event ERR: could not find primary "
		  << std::endl;
      }
      LogDebug("ForwardSim") << "CastorTestAnalysis: Particle Type " 
			     << particleType << " p/eta/phi " << pInit << ", "
			     << eta << ", " << phi;
  }

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10) 
    std::cout << " CastorTest Event " << iEvt << std::endl;
  else if ((iEvt < 100) && (iEvt%10 == 0)) 
    std::cout << " CastorTest Event " << iEvt << std::endl;
  else if ((iEvt < 1000) && (iEvt%100 == 0)) 
    std::cout << " CastorTest Event " << iEvt << std::endl;
  else if ((iEvt < 10000) && (iEvt%1000 == 0)) 
    std::cout << " CastorTest Event " << iEvt << std::endl;
						 
  std::cout << std::endl << "===>>> Done writing user histograms " << std::endl;
}

void CastorTestAnalysis::update(const EndOfRun * run) {;}
  
//=================================================================== 
void CastorTestAnalysis::getCastorBranchData(const CaloG4HitCollection * hc) {

    int nentries = hc->entries();
    
    if (nentries > 0) {
      
      unsigned int volumeID=0;
      int det=0, zside, sector, zmodule;
      std::map<int,float,std::less<int> > themap;
      double totalEnergy = 0;
      double hitEnergy = 0;
      double en_in_sd = 0.;

      for (int ihit = 0; ihit < nentries; ihit++) {
	CaloG4Hit* aHit = (*hc)[ihit];
	totalEnergy += aHit->getEnergyDeposit();
      }
    
      for (int ihit = 0; ihit < nentries; ihit++) {
	CaloG4Hit* aHit = (*hc)[ihit];
	volumeID = aHit->getUnitID();
	hitEnergy = aHit->getEnergyDeposit();
	en_in_sd += aHit->getEnergyDeposit();
//	double enEm = aHit->getEM();
//	double enHad = aHit->getHadr();
	
	themap[volumeID] += aHit->getEnergyDeposit();
	// int det, zside, sector, zmodule;
	theCastorNumScheme->unpackIndex(volumeID, zside, sector,zmodule);

	// det = 2 ;  //  det=2/3 for CAFI/CAPL
	
	castoreventarray[ntcastore_evt]       = (float)eventIndex;
//	castoreventarray[ntcastore_ihit]      = (float)ihit;
	castoreventarray[ntcastore_ihit]      = (float)eventGlobalHit;
	castoreventarray[ntcastore_detector]  = (float)det;
	castoreventarray[ntcastore_sector]    = (float)sector;
	castoreventarray[ntcastore_module]    = (float)zmodule;
	castoreventarray[ntcastore_enem]      = en_in_sd;
	castoreventarray[ntcastore_enhad]     = totalEnergy;
	castoreventarray[ntcastore_hitenergy] = hitEnergy;
	castoreventarray[ntcastore_x]         = aHit->getPosition().x();
	castoreventarray[ntcastore_y]         = aHit->getPosition().y();
	castoreventarray[ntcastore_z]         = aHit->getPosition().z();
//	castoreventarray[ntcastore_x]         = aHit->getEntry().x();
//	castoreventarray[ntcastore_y]         = aHit->getEntry().y();
//	castoreventarray[ntcastore_z]         = aHit->getEntry().z();
	
	castoreventntuple->Fill(castoreventarray);
	
	eventGlobalHit++ ;
      }
    } // nentries > 0
}

//=================================================================== 

void CastorTestAnalysis::Finish() {
  if (doNTcastorstep) {
    castorOutputStepFile->cd();
    castorstepntuple->Write();
    std::cout << "CastorTestAnalysis: Ntuple step  written" <<std::endl;
    castorOutputStepFile->Close();
    std::cout << "CastorTestAnalysis: Step file closed" << std::endl;
  }
  
   if (doNTcastorevent) {
   castorOutputEventFile->cd();
   castoreventntuple->Write("",TObject::kOverwrite);
   std::cout << "CastorTestAnalysis: Ntuple event written" << std::endl;   
   castorOutputEventFile->Close();
   std::cout << "CastorTestAnalysis: Event file closed" << std::endl;
 }

}        
