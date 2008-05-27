/* Analyize tau decays in various processes using tauola
*/

//
// Created:  Avto Kharchilava, Feb. 2008 
// Modified: 
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "TauAnalyzer.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"


TauAnalyzer::TauAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  invmass_histo = new TH1D("invmass_histo","invmass_histo",50,0,500);
  pT1_histo = new TH1D("pT1_histo","pT1_histo",50,0,500);
  pT2_histo = new TH1D("pT2_histo","pT2_histo",50,0,500);

  h_mcRtau = new TH1D("h_mcRtau","",20,0,1);
  h_mcRleptonic = new TH1D("h_mcRleptonic","",20,0,1);

  eventCounter         		 = 0;
  mcHadronicTauCounter 		 = 0;
  mcVisibleTauCounter  		 = 0;
  mcTauPtCutCounter    		 = 0;

}


TauAnalyzer::~TauAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  bool select = false;
  eventCounter++;

  bool lepton = false;

  math::XYZTLorentzVector LvectorTau(0,0,0,0);
  math::XYZTLorentzVector LvectorLep(0,0,0,0);
  math::XYZTLorentzVector visibleTau(0,0,0,0);
  math::XYZTLorentzVector leadingTrack(0,0,0,0);

  Handle<HepMCProduct> mcEventHandle;
  try{
    iEvent.getByLabel("source",mcEventHandle);
  }catch(...) {;}

  if(mcEventHandle.isValid()){
    const HepMC::GenEvent* mcEvent = mcEventHandle->GetEvent() ;

    HepMC::GenEvent::particle_const_iterator i;
    for(i = mcEvent->particles_begin(); i!= mcEvent->particles_end(); i++){
      int id = (*i)->pdg_id();

      if(abs(id) != 15) continue;
      
      
      int motherId  = 0;
      if( (*i)->production_vertex() ) {
	HepMC::GenVertex::particle_iterator iMother;
	for(iMother = (*i)->production_vertex()->particles_begin(HepMC::parents);
	    iMother!= (*i)->production_vertex()->particles_end(HepMC::parents); iMother++){
	  motherId = (*iMother)->pdg_id();

	  std::cout << " tau mother " <<  motherId   <<  std::endl;
	}
      }

      if( abs(motherId) != 37 ) continue;

      HepMC::FourVector p4 = (*i)->momentum();
      LvectorTau = math::XYZTLorentzVector(p4.px(),p4.py(),p4.pz(),p4.e());
      visibleTau = math::XYZTLorentzVector(p4.px(),p4.py(),p4.pz(),p4.e());

      if( (*i)->production_vertex() ) {
	HepMC::GenVertex::particle_iterator iChild;
	for(iChild = (*i)->production_vertex()->particles_begin(HepMC::descendants);
	    iChild!= (*i)->production_vertex()->particles_end(HepMC::descendants);iChild++){
	  int childId = (*iChild)->pdg_id();

	  std::cout << "tau child id " << childId << std::endl;

	  HepMC::FourVector fv = (*iChild)->momentum();
	  math::XYZTLorentzVector p(fv.px(),fv.py(),fv.pz(),fv.e());

	  if( abs(childId) == 12 || abs(childId) == 14 || abs(childId) == 16){
	    if((*iChild)->status() == 1 && childId*id > 0) {
	      visibleTau -= p;
	    }
	  }

	  if( abs(childId) == 11 || abs(childId) == 13 ){
	    lepton = true;
	    LvectorLep = p;

	    std::cout << "chiled lepton id " << childId << std::endl;

	  }
	  
	  if( abs(childId) == 211 ){ // pi+,rho+
	    if(p.P() > leadingTrack.P()) leadingTrack = p;
	  }
	  
	}
      }
      

    }
    
    
  }
  // ???  if(lepton) return select;
  if( lepton ) {

    double Rtau_lep = LvectorLep.E()/LvectorTau.E();
    h_mcRleptonic->Fill(Rtau_lep);
  }

  mcHadronicTauCounter++;
  
  // ???  if(visibleTau.Pt() == 0) return select;
  mcVisibleTauCounter++;
  std::cout << "vis tau px,py,pz,Pt " << visibleTau.Px() << " " << visibleTau.Py() << " " << visibleTau.Pz() << " " << visibleTau.Pt() << std::endl;
//std::cout << "        eta,phi  " << visibleTau.Eta() << " " << visibleTau.Phi() << std::endl;
  
  
// ???  if(visibleTau.Pt() < 100) return select;
  mcTauPtCutCounter++;

  std::cout << "visible tau pt " << visibleTau.Pt() << std::endl;

  if(!lepton && visibleTau.Pt() > 100.) {

    double Rtau = leadingTrack.P()/visibleTau.E();
    h_mcRtau->Fill(Rtau);
    std::cout << "check Rtau " << Rtau << " " << leadingTrack.P() << " " << visibleTau.E() << std::endl;
  }
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
TauAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TauAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  invmass_histo->Write();
  pT1_histo->Write();
  pT2_histo->Write();
  h_mcRtau->Write();
  h_mcRleptonic->Write();

  file.Close();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TauAnalyzer);
