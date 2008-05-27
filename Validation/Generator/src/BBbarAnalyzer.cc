/* This is en example for an Analyzer of a Herwig HepMCProduct
   and looks for muons pairs and fills a histogram
   with the invaraint mass of the four. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: BBbarAnalyzer.cc,v 1.2 2008/03/26 21:23:49 ksmith Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "BBbarAnalyzer.h"






#include "DataFormats/Math/interface/LorentzVector.h"



BBbarAnalyzer::BBbarAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  Pt_histo = new TH1F("invmass_histo","invmass_histo",100,0,20);
  invmass_histo = new TH1F("mu_invmass_histo","mu_invmass_histo",100,0,20);
}


BBbarAnalyzer::~BBbarAnalyzer()
{
 
}

// ------------ method called to for each event  ------------
void
BBbarAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
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

 Handle<HepMCProduct> mcEventHandle;
  try{
    iEvent.getByLabel("source",mcEventHandle);
  }catch(...) {;}

  if(mcEventHandle.isValid()){
    const HepMC::GenEvent* mcEvent = mcEventHandle->GetEvent() ;
   
   // if there are at least four muons
   // calculate invarant mass of first two and fill it into histogram
   math::XYZTLorentzVector tot_momentum;  math::XYZTLorentzVector tot_mumomentum; 
   float inv_mass = 0.0; double mu_invmass = 0.0; float Pt = 0; 
   
   HepMC::GenEvent::particle_const_iterator i;
   HepMC::GenEvent::particle_const_iterator j;
   for(i = mcEvent->particles_begin(); i!= mcEvent->particles_end(); i++){
     for(j = mcEvent->particles_begin(); j!= mcEvent->particles_end(); j++){
     HepMC::FourVector p41 = (*i)->momentum();
     HepMC::FourVector p42 = (*i)->momentum();
     Lvector1 = math::XYZTLorentzVector(p41.px(),p41.py(),p41.pz(),p41.e());
     Lvector2 = math::XYZTLorentzVector(p42.px(),p42.py(),p42.pz(),p42.e());
     Pt = sqrt(p41.px()*p41.px()+p41.py()*p41.py());
     Pt_histo->Fill(Pt);
     inv_mass = sqrt((p41.e()+p42.e())*(p41.e()+p42.e())-((p41.px()+p42.px())*(p41.px()+p42.px())+(p41.py()+p42.py())*(p41.py()+p42.py())+(p41.pz()+p42.pz())*(p41.pz()+p42.pz())));
     invmass_histo->Fill(inv_mass);
     }
   }
  }
}
     

// ------------ method called once each job just before starting event loop  ------------
void 
BBbarAnalyzer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BBbarAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  Pt_histo->Write();
  invmass_histo->Write();
  file.Close();

}

//define this as a plug-in
DEFINE_FWK_MODULE(BBbarAnalyzer);
