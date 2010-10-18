/* This is en example for an Analyzer of a Herwig HepMCProduct
   and looks for muons pairs and fills a histogram
   with the invaraint mass of the four. 
*/

//
// Original Author:  Fabian Stoeckli
//         Created:  Tue Nov 14 13:43:02 CET 2006
// $Id: H4muAnalyzer.cc,v 1.2 2010/01/06 12:39:47 fabstoec Exp $
//
//

// system include files
#include <memory>
#include <iostream>
// user include files
#include "Validation/Generator/plugins/H4muAnalyzer.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"


H4muAnalyzer::H4muAnalyzer(const edm::ParameterSet& iConfig)
{
  outputFilename=iConfig.getUntrackedParameter<std::string>("OutputFilename","dummy.root");
  invmass_histo = new TH1D("invmass_histo","invmass_histo",60,180,200);
  Z_invmass_histo = new TH1D("mu_invmass_histo","mu_invmass_histo",60,70,100);
}

H4muAnalyzer::~H4muAnalyzer()
{
 
}

void
H4muAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   // get HepMC::GenEvent ...
   Handle<HepMCProduct> evt_h;
   iEvent.getByType(evt_h);
   HepMC::GenEvent * evt = new  HepMC::GenEvent(*(evt_h->GetEvent()));


   // look for stable muons
   std::vector<HepMC::GenParticle*> muons;   
   muons.resize(0);
   for(HepMC::GenEvent::particle_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
     if(std::abs((*it)->pdg_id())==13 && (*it)->status()==1) {
       muons.push_back(*it);
     }
   }
   
   // if there are at least four muons
   // calculate invarant mass of first two and fill it into histogram
   math::XYZTLorentzVector tot_momentum;  math::XYZTLorentzVector tot_mumomentum; 
   double inv_mass = 0.0; double mu_invmass = 0.0;
   if(muons.size()>1) {
     for(unsigned int i=0; i<muons.size() - 1; ++i) {
       for(unsigned int j = i+1; j < muons.size(); ++j){
       math::XYZTLorentzVector mumom(muons[i]->momentum());
       math::XYZTLorentzVector mumom1(muons[j]->momentum());
       tot_mumomentum = mumom + mumom1;
       mu_invmass = sqrt(tot_mumomentum.mass2());
       Z_invmass_histo->Fill(mu_invmass);
       }
   }
     //invmass_histo->Fill(inv_mass);

}
 
   if(muons.size()>3) {
     for(unsigned int i=0; i<4; ++i) {
       math::XYZTLorentzVector mom(muons[i]->momentum());
       tot_momentum += mom;
     }
     inv_mass = sqrt(tot_momentum.mass2());
   }
   invmass_histo->Fill(inv_mass);

}

void 
H4muAnalyzer::beginJob()
{
}

void 
H4muAnalyzer::endJob() {
  // save histograms into file
  TFile file(outputFilename.c_str(),"RECREATE");
  invmass_histo->Write();
  Z_invmass_histo->Write();
  file.Close();

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(H4muAnalyzer);
