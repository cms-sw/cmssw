/* This is en example for an Analyzer of a Herwig HepMCProduct
   and looks for muons pairs and fills a histogram
   with the invaraint mass of the four. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: NewAnalyzer.cc,v 1.1 2008/05/23 17:02:43 ksmith Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "NewAnalyzer.h"
#include "DataFormats/JetReco/interface/GenJet.h"


#include <DataFormats/HepMCCandidate/interface/GenParticleCandidate.h>
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
{ /// Copy plots from Steve exactly 
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  J1Pt_histo = new TH1F("J1pT","J1pT",38,0,220);
  J2Pt_histo = new TH1F("J2pT","J2pT",38,0,120);
  E1Pt_histo = new TH1F("E1pT","E1pT",38,0,150);
  E2Pt_histo = new TH1F("E2pT","E2pT",38,0,150);
  ZPz_histo = new TH1F("ZPz","ZPz",100,0,20);
  ZPt_histo = new TH1F("ZPt","ZPt",38,0,200);
  J1Eta_histo = new TH1F("J1Eta_histo", "J1Eta_histo", 40, -3, 3); 
  J2Eta_histo = new TH1F("J2Eta_histo", "J2Eta_histo", 40, -3, 3);
  JDelR_histo = new TH1F("JDelR_histo", "JDelR_histo", 38, 0, 6);
  EJDelR_histo = new TH1F("EJDelR_histo", "EJDelR_histo", 38, 0, 6);
  JDelPhi_histo = new TH1F("JDelPhi_histo", "JDelPhi_histo", 38, 0, 3.2);
  J1Phi_histo = new TH1F("J1Phi_histo", "J1Phi_histo", 38, 0, 5); 
  J2Phi_histo = new TH1F("J2Phi_histo", "J2Phi_histo", 38, 0, 5);
  EE_invmass_histo = new TH1F("EE_invmass_histo","EE_invmass_histo",100,0,100);
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
 iEvent.getByLabel( "iterativeCone7GenJetsNoNuBSM", genJets);
 //Handle<GenJetCollection> genJets;
 //iEvent.getByLabel( "iterativeCone5GenJetsNoNuBSM", genJets);
 int elec = 0; 
 Handle<CandidateCollection> genPart;
  iEvent.getByLabel("genParticleCandidates",genPart);
  std::cout << "A" << std::endl;
  std::vector<float> elecEta; 
  std::vector<float> elecPhi;
  std::vector<float> elecPx; 
  std::vector<float> elecPy;
  std::vector<float> elecPz;
  float EEInvaMass;
  elecEta.clear();
  elecPhi.clear();
  elecPx.clear();
  elecPy.clear();
  elecPz.clear();
  float ptot, etot;
  math::XYZTLorentzVector MyJets1 (0,0,0,0);
  math::XYZTLorentzVector MyJets2 (0,0,0,0);
  //std::vector<math::XYZTLorentzVector> MyJets;
  for(size_t q = 0; q < genPart->size(); q++)
    {
      const Candidate & p = (*genPart)[q];
      int id = p.pdgId();
      size_t NMoth = p.numberOfMothers() ;
      int motherID1 = 0 ;
      int motherID2 = 0 ; 
      
      if(abs(id) == 23)
      {
        ZPt_histo->Fill(sqrt(p.px()*p.px() + p.py()*p.py()));
      }
      if(abs(id) != 11) continue;
      for ( size_t moth1=0; moth1<NMoth; moth1++ )
	{
	  motherID1 = (p.mother(moth1))->pdgId();
	  if(motherID1 == 23) 
	    {
	      elec++;
	      elecEta.push_back(p.eta());
	      elecPhi.push_back(p.phi());
	      elecPx.push_back(p.px());
	      elecPy.push_back(p.py());
	      elecPz.push_back(p.pz());
	    }
	}
      std::cout << "B" << std::endl;
      //if(abs(id) == 11 && sqrt(p.px()*p.px()+p.py()*p.py()) > 20 )  elec++;
    }
  if (elec > 1) 
    { 
      for(size_t elec1 = 0; elec1 < elec-1; elec1++)
	{
	  //const Candidate & p1 = (*genPart)[elec1];
	  //int id1 = p1.pdgId();
	  //int motherId1 = p1.mother()->pdgId;
	  //if(abs(id1) != 11) continue;
	  //TLorentzVector momentum = p1.momentum();
	  //cout <<" PT = " << sqrt(p1.px()*p1.px() +p1.py()*p1.py()) <<  endl;
	  for(size_t elec2 = elec1; elec2 < elec; elec2++)
	    {
	      //const Candidate & p2 = (*genPart)[elec2];
	      //int id2 = p2.pdgId();
	      //int motherId2 = p2->mother(0);
	      std::cout << "C1" << std::endl;
	      etot = sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]);
	      std::cout << "yay" << std::endl;
	      ptot =  sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2]) * (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2]) * (elecPz[elec1] + elecPz[elec2]));
	      std::cout << "yay2" << std::endl;
	      EEInvaMass = sqrt((etot+ptot)*(etot-ptot));
	      //EEInvaMass = sqrt(sqrt(elecPx[elec1]*elecPx[elec1]+elecPy[elec1]*elecPy[elec1]+elecPz[elec1]*elecPz[elec1])+sqrt(elecPx[elec2]*elecPx[elec2]+elecPy[elec2]*elecPy[elec2]+elecPz[elec2]*elecPz[elec2]) - sqrt((elecPx[elec1] + elecPx[elec2])* (elecPx[elec1] + elecPx[elec2]) + (elecPy[elec1] + elecPy[elec2])* (elecPy[elec2] + elecPy[elec1]) + (elecPz[elec1] + elecPz[elec2])* (elecPz[elec1] + elecPz[elec2]))) ; 
	      std::cout << "C2" << std::endl;
	      //EE_invmass_histo->Fill(EEInvaMass);
	      std::cout << "C3" << std::endl;	
	      //if(abs(id2) != 11) continue;
	      
	    }
	}

     std::cout << "C" << std::endl;
      if(genJets.isValid()){
	if(genJets->size() > 1)
	  {
	    int nmyJets = 0;
	    for(int Jets = 0; Jets < genJets->size(); Jets++)
	      {
		int incone = 0;
		const Candidate & J1 = (*genJets)[Jets];
		for(int elecs = 0; elecs < elecPhi.size(); elecs++)
		  {
		    float EJDelPhi = fabs(J1.phi()-elecPhi[elecs]);
		    if(EJDelPhi >  3.1415926) EJDelPhi = 6.2831852 - EJDelPhi;
		    float EJDelR = sqrt((J1.eta()-elecEta[elecs])*(J1.eta()-elecEta[elecs])+EJDelPhi*EJDelPhi);
		   
		    if (EJDelR < .2) {  cout << EJDelR << endl; incone++;}
		    if(elecs == elecPhi.size()) continue;
		    /*for(int elecs1 = elecs+1; elecs1 < elecPhi.size(); elecs1++)
		      {
			
			if(elecs == elecs1) continue;
				    
			}*/
		  }
		//if(nmyJets > 1) continue;
		if (incone == 0 && nmyJets == 0) MyJets1 = math::XYZTLorentzVector(J1.px(),J1.py(),J1.pz(),0.);///// ***** Filled e with 0
		if (incone == 0 && nmyJets == 1) MyJets2 = math::XYZTLorentzVector(J1.px(),J1.py(),J1.pz(),0.);
		nmyJets++;
		
		
	      }
	    std::cout << "D" << std::endl;
	    if(nmyJets >= 2)
	      {
		
		//for(int J1int = 0; J1int < genJets->size()-1 ; J1int++)
		//{
		//const Candidate & J1 = (*genJets)[J1int];
		//const Candidate & J1 = (*genJets)[0];
		double PtJ1 = MyJets1.pt();
		//float PtJ1 = sqrt(J1.px()*J1.px() + J1.py()*J1.py());
		float J1Eta = MyJets1.eta(); 
		float J1Phi = MyJets1.phi();
		if(PtJ1 > 20)
		  { 
		    //const Candidate & Temp = (*genJets)[J1int + 1 ];
		    //if(sqrt(Temp.px()*Temp.px()+Temp.py()*Temp.py()))
		    // {
		    J1Eta_histo->Fill(J1Eta);
		    J1Pt_histo->Fill(PtJ1);
		    J1Phi_histo->Fill(J1Phi);
		    //  }
		    //for(int J2int = J1int; J2int < genJets->size() ; J2int++)
		    // {
		    //const Candidate & J2 = (*genJets)[1];
			//const Candidate & J2 = (*genJets)[J2int];
		    double PtJ2 = MyJets2.pt();
		    //float PtJ2 = sqrt(J2.px()*J2.px() + J2.py()*J2.py());
			if(PtJ2 > 20)
			  {
			    float J2Eta = MyJets2.eta(); 
			    float J2Phi = MyJets2.phi();
			    J2Eta_histo->Fill(J2Eta);
			    J2Pt_histo->Fill(PtJ2);
			    float DelPhi = fabs(J1Phi-J2Phi); 
			    if(DelPhi >  3.1415926) DelPhi = 6.2831852 - DelPhi;
			    JDelPhi_histo->Fill(DelPhi);
			    float DelR = sqrt((J2Eta - J1Eta)*(J2Eta - J1Eta)+DelPhi*DelPhi);
			    JDelR_histo->Fill(DelR);
			    J2Phi_histo->Fill(J2Phi);
			  }
		  }
		
	      }  
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
   TFile file("Test.root","RECREATE");
  J1Pt_histo->Write();
  J2Pt_histo->Write();
  EE_invmass_histo->Write();
  ZPt_histo->Write();
  JDelR_histo->Write();
  JDelPhi_histo->Write();
  J1Eta_histo->Write();
  J2Eta_histo->Write();
  J1Phi_histo->Write();
  J2Phi_histo->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(NewAnalyzer);
