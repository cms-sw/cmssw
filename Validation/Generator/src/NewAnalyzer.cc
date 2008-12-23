/* This is en example for an Analyzer of a Herwig HepMCProduct
   and looks for muons pairs and fills a histogram
   with the invaraint mass of the four. 
*/

//
// Original Author:  Kenneth Smith
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: NewAnalyzer.cc,v 1.3 2008/07/03 21:04:52 ksmith Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "NewAnalyzer.h"
#include "DataFormats/JetReco/interface/GenJet.h"


#include <DataFormats/HepMCCandidate/interface/GenParticleCandidate.h>
#include <DataFormats/HepMCCandidate/interface/GenParticles.h>
#include <DataFormats/Candidate/interface/Candidate.h>

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "TLorentzVector.h"

NewAnalyzer::NewAnalyzer(const edm::ParameterSet& iConfig)
{ 
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  Jetmult_histo = new TH1F("Jetmult_histo","Jet multiplicity",5,0,5);
  J1Pt_histo = new TH1F("J1pT","J1 pT",38,0,220);
  J2Pt_histo = new TH1F("J2pT","J2 pT",38,0,120);
  JetPt1J = new TH1F("JetpT1J","Jet pT for 1 jet events",38,0,220);
  JetPt2J = new TH1F("JetpT2J","Jet pT for 2 jet events",38,0,220);
  JetPt3J = new TH1F("JetpT3J","Jet pT for 3 jet events",38,0,220);
  JetPt4J = new TH1F("JetpT4J","Jet pT for 4 jet events",38,0,220);
  Z1JJ1Pt_histo = new TH1F("Z1JJ1pT","Jet1 pT for Z+1J",38,0,220);
  Z2JJ1Pt_histo = new TH1F("Z2JJ1pT","Jet1 pT for Z+2J",38,0,220);
  Z2JJ2Pt_histo = new TH1F("Z2JJ2pT","Jet2 pT for Z+2J",38,0,120);
  Z3JJ1Pt_histo = new TH1F("Z3JJ1pT","Jet1 pT for Z+3J",38,0,220);
  Z3JJ2Pt_histo = new TH1F("Z3JJ2pT","Jet2 pT for Z+3J",38,0,120);
  Z4JJ1Pt_histo = new TH1F("Z4JJ1pT","Jet1 pT for Z+4J",38,0,220);
  Z4JJ2Pt_histo = new TH1F("Z4JJ2pT","Jet2 pT for Z+4J",38,0,120);
  E1Pt_histo = new TH1F("E1pT","Electron pT",38,0,150);
  E2Pt_histo = new TH1F("E2pT","E2 pT",38,0,150);
  ZPz_histo = new TH1F("ZPz","ZPz",100,0,20);
  ZPt_histo = new TH1F("ZPt_histo","Z Pt",38,0,200);
  ZPt1J_histo = new TH1F("ZPt1J","Z Pt Z+1J",38,0,200);
  ZPt2J_histo = new TH1F("ZPt2J","Z Pt Z+2J",38,0,200);
  ZPt3J_histo = new TH1F("ZPt3J","Z Pt Z+3J",38,0,200);
  ZPt4J_histo = new TH1F("ZPt4J","Z Pt Z+4J",38,0,200);
  ZPt0J_histo = new TH1F("ZPt0J","Z Pt Z+0J",38,0,200);
  J1Eta_histo = new TH1F("J1Eta_histo", "J1Eta_histo", 40, -3, 3);
  Z1JJ1Eta_histo = new TH1F("Z1JJ1Eta_histo", "Z1JJ1Eta_histo", 40, -3, 3); 
  Z2JJ1Eta_histo = new TH1F("Z2JJ1Eta_histo", "Z2JJ1Eta_histo", 40, -3, 3); 
  Z2JJ2Eta_histo = new TH1F("Z2JJ2Eta_histo", "Z2JJ2Eta_histo", 40, -3, 3);
  Z3JJ1Eta_histo = new TH1F("Z3JJ1Eta_histo", "Z3JJ1Eta_histo", 40, -3, 3); 
  Z3JJ2Eta_histo = new TH1F("Z3JJ2Eta_histo", "Z3JJ2Eta_histo", 40, -3, 3);
  Z4JJ1Eta_histo = new TH1F("Z4JJ1Eta_histo", "Z4JJ1Eta_histo", 40, -3, 3); 
  Z4JJ2Eta_histo = new TH1F("Z4JJ2Eta_histo", "Z4JJ2Eta_histo", 40, -3, 3);
  ZEta_histo =  new TH1F("ZEta_histo", "Z Eta", 40, -3, 3);
  ZRap_histo =  new TH1F("ZRap_histo", "Z Rapidity", 40, -3, 3);
  JDelR_histo = new TH1F("JDelR_histo", "JDelR_histo", 38, 0, 6);
  Z2JJDelR_histo = new TH1F("Z2JJDelR_histo", "Z2JJDelR_histo", 38, 0, 6);
  Z3JJDelR_histo = new TH1F("Z3JJDelR_histo", "Z3JJDelR_histo", 38, 0, 6);
  Z4JJDelR_histo = new TH1F("Z4JJDelR_histo", "Z4JJDelR_histo", 38, 0, 6);
  EJDelR_histo = new TH1F("EJDelR_histo", "EJDelR_histo", 38, 0, 6);
  JDelPhi_histo = new TH1F("JDelPhi_histo", "JDelPhi_histo", 38, -3.2, 3.2);
  Z2JJDelPhi_histo = new TH1F("Z2JJDelPhi_histo", "Z2JJDelPhi_histo", 38, -3.2, 3.2);
  Z3JJDelPhi_histo = new TH1F("Z3JJDelPhi_histo", "Z3JJDelPhi_histo", 38, -3.2, 3.2);
  Z4JJDelPhi_histo = new TH1F("Z4JJDelPhi_histo", "Z4JJDelPhi_histo", 38, -3.2, 3.2);
  J1Phi_histo = new TH1F("J1Phi_histo", "J1Phi_histo", 38, 0, 5); 
  Z1JJ1Phi_histo = new TH1F("Z1JJ1Phi_histo", "Z1JJ1Phi_histo", 38, 0, 5); 
  Z2JJ1Phi_histo = new TH1F("Z2JJ1Phi_histo", "Z2JJ1Phi_histo", 38, 0, 5); 
  Z2JJ2Phi_histo = new TH1F("Z2JJ2Phi_histo", "Z2JJ2Phi_histo", 38, 0, 5);
  Z3JJ1Phi_histo = new TH1F("Z3JJ1Phi_histo", "Z3JJ1Phi_histo", 38, 0, 5); 
  Z3JJ2Phi_histo = new TH1F("Z3JJ2Phi_histo", "Z3JJ2Phi_histo", 38, 0, 5);
  Z4JJ1Phi_histo = new TH1F("Z4JJ1Phi_histo", "Z4JJ1Phi_histo", 38, 0, 5); 
  Z4JJ2Phi_histo = new TH1F("Z4JJ2Phi_histo", "Z4JJ2Phi_histo", 38, 0, 5);
  Z_invmass_histo = new TH1F("Z_invmass_histo","Z_invmass_histo",200,0,200);
  Z0J_invmass_histo = new TH1F("Z0J_invmass_histo","Z0J_invmass_histo",200,0,200);
  Z1J_invmass_histo = new TH1F("Z1J_invmass_histo","Z1J_invmass_histo",200,0,200);
  Z2J_invmass_histo = new TH1F("Z2J_invmass_histo","Z2J_invmass_histo",200,0,200);
  Z3J_invmass_histo = new TH1F("Z3J_invmass_histo","Z3J_invmass_histo",200,0,200);
  Z4J_invmass_histo = new TH1F("Z4J_invmass_histo","Z4J_invmass_histo",200,0,200);
  int event = 0 ; 
}


NewAnalyzer::~NewAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
NewAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ++event;
   using namespace edm;
   using namespace std;
   using namespace reco;
   // get HepMC::GenEvent ...
   //Handle<HepMCProduct> evt_h;
   //iEvent.getByType(evt_h);
   //HepMC::GenEvent * evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));


   // look for stable muons
   // std::vector<HepMC::GenParticle*> muons;   
   //muons.resize(0);
   //for(HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
   // if(abs((*it)->pdg_id())==13 && (*it)->status()==1) {
   //   muons.push_back(*it);
   // }
   // }

  math::XYZTLorentzVector Lvector1(0,0,0,0);
  math::XYZTLorentzVector Lvector2(0,0,0,0);

  // Handle<HepMCProduct> mcEventHandle;
  //try{
  //iEvent.getByLabel("source",mcEventHandle);
  //}catch(...) {;}
  typedef std::vector<reco::GenJet> GenJetCollection;
 Handle<GenJetCollection> genJets;
 //iEvent.getByLabel( "iterativeCone5GenJetsNoNuBSM", genJets);
 //Handle<GenJetCollection> genJets;
 iEvent.getByLabel( "iterativeCone5GenJets", genJets);
 int elec = 0; 
 //std::cout << "A" << std::endl;
//Handle<CandidateCollection> genPart;
//iEvent.getByLabel("genParticleCandidates",genPart);

 Handle<GenParticleCollection> genPart;
  iEvent.getByLabel("genParticles",genPart);
  //std::cout << "A" << std::endl;
  std::vector<float> elecEta; 
  std::vector<float> elecPhi;
  std::vector<float> elecPx; 
  std::vector<float> elecPy;
  std::vector<float> elecPz;
  std::vector<int> elecCh;
  std::vector<double> JetpT;
  float EEInvaMass;
  elecEta.clear();
  elecPhi.clear();
  elecPx.clear();
  elecPy.clear();
  elecPz.clear();
  double ZpT;
  float ptot, etot;
  double Jet1Pt, Jet2Pt;
  Jet1Pt = 0; 
  Jet2Pt = 0; 
  math::XYZTLorentzVector MyJets1 (0,0,0,0);
  math::XYZTLorentzVector MyJets2 (0,0,0,0);
  for(size_t q = 0; q < genPart->size(); q++)
    {
      const Candidate & p = (*genPart)[q];
      int id = p.pdgId();
      size_t NMoth = p.numberOfMothers() ;
      int motherID1 = 0 ;
      int motherID2 = 0 ; 
      if(abs(id) == 23)
	{
	  //ZPt_histo->Fill(sqrt(p.px()*p.px() + p.py()*p.py()));
	  ZEta_histo->Fill(p.eta());
	  ZRap_histo->Fill(p.rapidity());
	  ZpT = p.pt();
	}
      if(abs(id) != 11) continue;
      for ( size_t moth1=0; moth1<NMoth; moth1++ )
	{
	  motherID1 = (p.mother(moth1))->pdgId();
	  if(fabs(motherID1) == 23) 
	    {
	      elec++;
	      elecEta.push_back(p.eta());
	      elecPhi.push_back(p.phi());
	      elecPx.push_back(p.px());
	      elecPy.push_back(p.py());
	      elecPz.push_back(p.pz());
	      elecCh.push_back(p.charge());
	      E1Pt_histo->Fill(sqrt(p.px()*p.px() + p.py() * p.py()));
	    }
	}
      
    }
  if (elec > 1) 
    { 
      ZPt_histo->Fill(ZpT);
      if(genJets.isValid()){
	if(genJets->size() > 1)
	  {
	   
	    for(size_t elec1 = 0; elec1 < elec-1; elec1++)
	      {
		for(size_t elec2 = elec1 + 1; elec2 < elec; elec2++)
		  {
		    if(elecCh[elec2] == elecCh[elec1])
		      continue;
		    etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
		    ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
		    EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
		    Z_invmass_histo->Fill(EEInvaMass);
		  }
	      }
	    int nmyJets = 0;
	    JetpT.clear();
	    //Jet1Pt = 0.0;
	    //Jet2Pt = 0.0;
	    for(int Jets = 0; Jets < genJets->size(); Jets++)
	      {
		int incone = 0;
		const Candidate & J1 = (*genJets)[Jets];
		for(int elecs = 0; elecs < elec; elecs++)
		  {
		    float EJDelPhi = fabs(J1.phi()-elecPhi[elecs]);
		    if(EJDelPhi >  3.1415926) EJDelPhi = 6.2831852 - EJDelPhi;
		    float EJDelR = sqrt((J1.eta()-elecEta[elecs])*(J1.eta()-elecEta[elecs])+EJDelPhi*EJDelPhi);
		   
		    if (EJDelR < .2) {  //cout << EJDelR << endl; 
		      incone++;}
		    if(elecs == elecPhi.size()) continue;
		  }
		//cout << J1.pt() << " Jet pT " << endl;
		
		if (incone == 0 && J1.pt() > 12)
		  {
		    if(nmyJets == 0 ) 
		      {
			MyJets1 = math::XYZTLorentzVector(J1.px(),J1.py(),J1.pz(),J1.energy());
			Jet1Pt = J1.pt();
			
			//cout << "Jet 1 found " << endl;
		      }
		    if ( nmyJets == 1)
		      {
			MyJets2 = math::XYZTLorentzVector(J1.px(),J1.py(),J1.pz(),J1.energy());
			Jet2Pt = J1.pt();
			//cout << "Jet 2 found " << endl;
		      }
		    nmyJets++;
		    JetpT.push_back(J1.pt());
		  }
		
	      }
	    if(JetpT.size() == 1)
	      for(int i = 0; i < JetpT.size(); i++)
		{
		  JetPt1J->Fill(JetpT[i]);
		}
	    if(JetpT.size() == 2)
	      for(int i = 0; i < JetpT.size(); i++)
		{
		  JetPt2J->Fill(JetpT[i]);
		}
	    if(JetpT.size() == 3)
	      for(int i = 0; i < JetpT.size(); i++)
		{
		  JetPt3J->Fill(JetpT[i]);
		}
	    if(JetpT.size() > 3)
	      for(int i = 0; i < JetpT.size(); i++)
		{
		  JetPt4J->Fill(JetpT[i]);
		}
	    //std::cout << nmyJets << " my Jets" << std::endl;
	    //std::cout << "Jet 1 pt " << Jet1Pt << " Jet 2 Pt " << Jet2Pt <<  std::endl;
	    if(nmyJets == 0 || (Jet1Pt < 12 && Jet2Pt < 12))
	      {
		for(size_t elec1 = 0; elec1 < elec-1; elec1++)
		  {
		    for(size_t elec2 = elec1 + 1; elec2 < elec; elec2++)
		      {
			if(elecCh[elec2] == elecCh[elec1])
			  continue;
			etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1]) + 
			       sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
			ptot =  sqrt((elecPx[elec1] + elecPx[elec2]) * (elecPx[elec1] + elecPx[elec2]) + 
				     (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + 
				     (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
			EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
			Z0J_invmass_histo->Fill(EEInvaMass);
			ZPt0J_histo->Fill(ZpT);
		      }
		  }
		Jetmult_histo->Fill(0);
	      }
	    if(nmyJets > 0 )
	      {
		double PtJ1 = MyJets1.pt();
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(Jet1Pt > 12)
		  { 
		    J1Pt_histo->Fill(PtJ1);
		    J1Phi_histo->Fill(J1Phi);
		    J1Eta_histo->Fill(J1Eta);
		  }
	      }
	    if(nmyJets == 1 )
	      {
		Jetmult_histo->Fill(1);
		double PtJ1 = MyJets1.pt();
		cout << Jet1Pt << " Jet Pt" << endl;
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(Jet1Pt > 12)
		  { 
		    Z1JJ1Eta_histo->Fill(J1Eta);
		    Z1JJ1Pt_histo->Fill(PtJ1);
		    Z1JJ1Phi_histo->Fill(J1Phi);
		    ZPt1J_histo->Fill(ZpT);
		  }
		for(size_t elec1 = 0; elec1 < elec-1; elec1++)
		  {
		    for(size_t elec2 = elec1 + 1; elec2 < elec; elec2++)
		      {
			if(elecCh[elec2] == elecCh[elec1])
			  continue;
			etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
			ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
			EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
			Z1J_invmass_histo->Fill(EEInvaMass);
		      }
		  }
	      }
	    if(nmyJets == 2)
	      {
		Jetmult_histo->Fill(2);
		double PtJ1 = MyJets1.pt();
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(Jet1Pt > 12)
		  { 
		    Z2JJ1Eta_histo->Fill(J1Eta);
		    Z2JJ1Pt_histo->Fill(PtJ1);
		    Z2JJ1Phi_histo->Fill(J1Phi);
		    double PtJ2 = MyJets2.pt();
			if(Jet2Pt > 12)
			  {
			    float J2Eta = MyJets2.eta(); 
			    float J2Phi = MyJets2.phi();
			    Z2JJ2Eta_histo -> Fill(J2Eta);
			    Z2JJ2Pt_histo -> Fill(PtJ2);
			    float DelPhi = fabs(J1Phi-J2Phi); 
			    if(DelPhi >  3.1415926) DelPhi = 6.2831852 - DelPhi;
			    Z2JJDelPhi_histo -> Fill(DelPhi);
			    float DelR = sqrt((J2Eta - J1Eta)*(J2Eta - J1Eta)+DelPhi*DelPhi);
			    Z2JJDelR_histo -> Fill(DelR);
			    Z2JJ2Phi_histo -> Fill(J2Phi);
			    ZPt2J_histo->Fill(ZpT);
			  }
		  }
		for(size_t elec1 = 0; elec1 < elec-1; elec1++)
		  {
		    for(size_t elec2 = elec1 +1 ; elec2 < elec; elec2++)
		      {
			if(elecCh[elec2] == elecCh[elec1])
			  continue;
			etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
			ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
			EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
			Z2J_invmass_histo->Fill(EEInvaMass);
	      
		      }
		  }
	      }
	    if(nmyJets == 3)
	      {
		Jetmult_histo->Fill(3);
		double PtJ1 = MyJets1.pt();
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(Jet1Pt > 12)
		  { 
		    Z3JJ1Eta_histo->Fill(J1Eta);
		    Z3JJ1Pt_histo->Fill(PtJ1);
		    Z3JJ1Phi_histo->Fill(J1Phi);
		    double PtJ2 = MyJets2.pt();
			if(Jet2Pt > 12)
			  {
			    float J2Eta = MyJets2.eta(); 
			    float J2Phi = MyJets2.phi();
			    Z3JJ2Eta_histo->Fill(J2Eta);
			    Z3JJ2Pt_histo->Fill(PtJ2);
			    float DelPhi = fabs(J1Phi-J2Phi); 
			    if(DelPhi >  3.1415926) DelPhi = 6.2831852 - DelPhi;
			    Z3JJDelPhi_histo->Fill(DelPhi);
			    float DelR = sqrt((J2Eta - J1Eta)*(J2Eta - J1Eta)+DelPhi*DelPhi);
			    Z3JJDelR_histo->Fill(DelR);
			    Z3JJ2Phi_histo->Fill(J2Phi);
			    ZPt3J_histo->Fill(ZpT);
			  }
		  }
		for(size_t elec1 = 0; elec1 < elec-1; elec1++)
		  {
		    for(size_t elec2 = elec1 +1 ; elec2 < elec; elec2++)
		      {	
			if(elecCh[elec2] == elecCh[elec1])
			  continue;
			etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
			ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
			EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
			Z3J_invmass_histo->Fill(EEInvaMass);
	      
		      }
		  }
	      }
	    if(nmyJets > 3)
	      {
		Jetmult_histo->Fill(4);
		//std::cout << "C7" << std::endl;
		double PtJ1 = MyJets1.pt();
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(Jet1Pt > 12)
		  { 
		    //std::cout << "C8" << std::endl;
		    Z4JJ1Eta_histo->Fill(J1Eta);
		    Z4JJ1Pt_histo->Fill(PtJ1);
		    Z4JJ1Phi_histo->Fill(J1Phi);
		    double PtJ2 = MyJets2.pt();
		    //std::cout << "C9" << std::endl;
			if(Jet2Pt > 12)
			  {
			    float J2Eta = MyJets2.eta(); 
			    float J2Phi = MyJets2.phi();
			    //std::cout << "C10" << std::endl;
			    Z4JJ2Eta_histo->Fill(J2Eta);
			    Z4JJ2Pt_histo->Fill(PtJ2);
			    float DelPhi = fabs(J1Phi-J2Phi); 
			    if(DelPhi >  3.1415926) DelPhi = 6.2831852 - DelPhi;
			    Z4JJDelPhi_histo->Fill(DelPhi);
			    float DelR = sqrt((J2Eta - J1Eta)*(J2Eta - J1Eta)+DelPhi*DelPhi);
			    Z4JJDelR_histo->Fill(DelR);
			    Z4JJ2Phi_histo->Fill(J2Phi);
			    ZPt4J_histo->Fill(ZpT);
			  }
		  }
		for(size_t elec1 = 0; elec1 < elec-1; elec1++)
		  {
		    for(size_t elec2 = elec1+1; elec2 < elec; elec2++)
		      {
			if(elecCh[elec2] == elecCh[elec1])
			  continue;
			etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
			ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
			EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
			Z4J_invmass_histo->Fill(EEInvaMass);
	      
		      }
		  }
	      }
	  }
	else
	  {
	    ZPt0J_histo->Fill(ZpT);
	  }
	
      std::cout << "E" << std::endl;
      }
      
  // if there are at least four muons
      // calculate invarant mass of first two and fill it into histogram
      math::XYZTLorentzVector tot_momentum;  math::XYZTLorentzVector tot_mumomentum; 
      float inv_mass = 0.0; double mu_invmass = 0.0; float Pt = 0; 
      for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end() ;  ++gen ) 
	{
	  //cout << "gen jet pt " << gen->pt() << endl ;   
	}
	
    }

  //cout << elec << event << endl;
}

     

// ------------ method called once each job just before starting event loop  ------------
void 
NewAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
NewAnalyzer::endJob() {
  // save histograms into file
   TFile file("ZJets__MG.root","RECREATE");
  J1Pt_histo->Write();
  J2Pt_histo->Write();
  Z_invmass_histo->Write();
  Z1J_invmass_histo->Write();
  Z2J_invmass_histo->Write();
  Z3J_invmass_histo->Write();
  Z4J_invmass_histo->Write();
  ZPt_histo->Write();
  JDelR_histo->Write();
  JDelPhi_histo->Write();
  Z1JJ1Eta_histo->Write();
  ZEta_histo->Write();
  Z1JJ1Phi_histo->Write();
  Z2JJ1Eta_histo->Write();
  Z2JJ2Eta_histo->Write();
  Z2JJ1Phi_histo->Write();
  Z2JJ2Phi_histo->Write();
  Z3JJ1Eta_histo->Write();
  Z3JJ2Eta_histo->Write();
  Z3JJ1Phi_histo->Write();
  Z3JJ2Phi_histo->Write();
  Z4JJ1Eta_histo->Write();
  Z4JJ2Eta_histo->Write();
  Z4JJ1Phi_histo->Write();
  Z4JJ2Phi_histo->Write();
  Jetmult_histo->Write();
  Z2JJDelR_histo->Write();
  Z2JJ2Phi_histo->Write();
  Z3JJDelR_histo->Write();
  Z3JJ2Phi_histo->Write();
  Z4JJDelR_histo->Write();
  Z4JJ2Phi_histo->Write();
  JetPt1J->Write();
  JetPt2J->Write();
  JetPt3J->Write();
  JetPt4J->Write();
  ZRap_histo->Write();
  ZPt0J_histo->Write();
  ZPt1J_histo->Write();
  ZPt2J_histo->Write();
  ZPt3J_histo->Write();
  ZPt4J_histo->Write();
  Z0J_invmass_histo->Write();
  Z1J_invmass_histo->Write();
  Z2J_invmass_histo->Write();
  Z3J_invmass_histo->Write();
  Z4J_invmass_histo->Write();
  E1Pt_histo->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(NewAnalyzer);
