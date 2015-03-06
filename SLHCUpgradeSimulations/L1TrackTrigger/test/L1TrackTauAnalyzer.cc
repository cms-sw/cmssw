// -*- C++ -*-
//
// Package:    L1TrackTriggerObjectsAnalyzer
// Class:      L1TrackTriggerObjectsAnalyzer
// 
/**\class L1TrackTriggerObjectsAnalyzer L1TrackTriggerObjectsAnalyzer.cc SLHCUpgradeSimulations/L1TrackTriggerObjectsAnalyzer/src/L1TrackTriggerObjectsAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Thu Nov 14 11:22:13 CET 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"


// Gen-level stuff:
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"


#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


using namespace l1extra;



//
// class declaration
//

class L1TrackTauAnalyzer : public edm::EDAnalyzer {
public:
  
  typedef L1TkTrack_PixelDigi_                          L1TkTrackType;
  typedef std::vector< L1TkTrackType >                               L1TkTrackCollectionType;
  
  explicit L1TrackTauAnalyzer(const edm::ParameterSet&);
  ~L1TrackTauAnalyzer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  //virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  //virtual void endRun(edm::Run const&, edm::EventSetup const&);
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void getStableDaughters(const reco::Candidate & p, 
			  std::vector<const reco::Candidate *>& stabledaughters);

  int tauClass(std::vector<const reco::Candidate *>& stabledaughters);

  
  // ----------member data ---------------------------

  
  TTree* eventTree;

  //reconstructed taus
  std::vector<float>* m_tau_pt;
  std::vector<float>* m_tau_eta;
  std::vector<float>* m_tau_phi;
  std::vector<float>* m_tau_z;
  
  //generated taus
  std::vector<float>* m_taugen_pt;
  std::vector<float>* m_taugen_eta;
  std::vector<float>* m_taugen_phi;
  std::vector<float>* m_taugen_etvis;
  std::vector<float>* m_taugen_z;
  std::vector<int>* m_taugen_class; 


  // for L1TkTauParticles
  edm::InputTag L1TkTauInputTag;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1TrackTauAnalyzer::L1TrackTauAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed


  L1TkTauInputTag = iConfig.getParameter<edm::InputTag>("L1TkTauInputTag");

}


L1TrackTauAnalyzer::~L1TrackTauAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

int L1TrackTauAnalyzer::tauClass(std::vector<const reco::Candidate *>& stabledaughters) {

  // -999 means not classified
  // 1 means electron
  // 2 means muon
  // 3 means had 1 prong
  // 4 means had 3 prong
  // 5 means had 5 prong

  if (stabledaughters[1]->pdgId()==11||stabledaughters[1]->pdgId()==-11) return 1;
  if (stabledaughters[1]->pdgId()==13||stabledaughters[1]->pdgId()==-13) return 2;
    
  int nprong=0;  

  for (unsigned int i=0;i<stabledaughters.size();i++) {
    if (stabledaughters[i]->pdgId()==211||stabledaughters[i]->pdgId()==-211) nprong++;
  }

  if (nprong==1) return 3;
  if (nprong==3) return 4;
  if (nprong==5) return 5;

  return -999;

}


void L1TrackTauAnalyzer::getStableDaughters(const reco::Candidate & p, 
			std::vector<const reco::Candidate *>& stabledaughters){

  int ndaug=p.numberOfDaughters();
  
  for(int j=0;j<ndaug;j++){
    //double vx = p.vx(), vy = p.vy(), vz = p.vz();

    //std::cout << "daug vertex : "<<vx<<" "<<vy<<" "<<vz<<std::endl;

    const reco::Candidate * daug = p.daughter(j);
    if (daug->status()==1) {
      stabledaughters.push_back(daug);
    }
    else {
      getStableDaughters(*daug,stabledaughters);
    }
  }  

}

// ------------ method called for each event  ------------
void
L1TrackTauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   m_tau_pt->clear();
   m_tau_eta->clear();
   m_tau_phi->clear();
   m_tau_z->clear();
  
   m_taugen_pt->clear();
   m_taugen_eta->clear();
   m_taugen_phi->clear();
   m_taugen_etvis->clear();
   m_taugen_z->clear();
   m_taugen_class->clear();


   //std::cout << " ----  a new event ----- " << std::endl;

   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   for(size_t i = 0; i < genParticles->size(); ++ i) {
     const GenParticle & p = (*genParticles)[i];
     int id = p.pdgId();
     //int st = p.status();
     double eta=p.p4().eta();
     double phi=p.p4().phi();
     double pt=p.p4().pt();
     if (!(id==15||id==-15)) continue;
     int nmother=p.numberOfMothers();
     if (nmother!=1) continue;
     if (abs(p.mother(0)->pdgId())==15) continue;

     double vz = p.vz();

     std::vector<const reco::Candidate *> stabledaughters;

     getStableDaughters(p,stabledaughters);
     int tauclass=tauClass(stabledaughters);

     double etvis=0.0;
     for(unsigned int j=0;j<stabledaughters.size();j++){
       if (abs(stabledaughters[j]->pdgId())==16) continue;
       etvis+=stabledaughters[j]->p4().pt();
       //std::cout << "daug: "<<stabledaughters[j]<<std::endl;
     }



     m_taugen_pt->push_back(pt);
     m_taugen_eta->push_back(eta);
     m_taugen_phi->push_back(phi);
     m_taugen_etvis->push_back(etvis);
     m_taugen_z->push_back(vz);
     m_taugen_class->push_back(tauclass);

   
   }

   //
   // ----------------------------------------------------------------------
   // retrieve the L1TkTau objects
   //

   
   edm::Handle<L1TkTauParticleCollection> L1TkTausHandle;
   iEvent.getByLabel(L1TkTauInputTag, L1TkTausHandle);
   std::vector<L1TkTauParticle>::const_iterator tauIter ;
   

   if ( L1TkTausHandle.isValid() ) {
     //std::cout << " -----   L1TkTauParticle  objects -----  " << std::endl;
     for (tauIter = L1TkTausHandle -> begin(); tauIter != L1TkTausHandle->end(); ++tauIter) {
       float et = tauIter -> pt();
       float phi = tauIter -> phi();
       float eta = tauIter -> eta();
       float z = 0.0;

       if (!tauIter->getTrkPtr().isNull()) {
	 z=tauIter->getTrkPtr()->getPOCA().z();
       }

       m_tau_pt->push_back(et);
       m_tau_phi->push_back(phi);
       m_tau_eta->push_back(eta);
       m_tau_z->push_back(z);
     }
   } else {
     std::cout << "Did not find L1TkTauParticle collection"<<std::endl;
   }


   eventTree->Fill();

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TrackTauAnalyzer::beginJob()
{
  
  edm::Service<TFileService> fs;

  //reconstructed taus
  m_tau_pt = new std::vector<float>;
  m_tau_eta = new std::vector<float>;
  m_tau_phi = new std::vector<float>;
  m_tau_z = new std::vector<float>;
  
  //generated taus
  m_taugen_pt = new std::vector<float>;
  m_taugen_eta = new std::vector<float>;
  m_taugen_phi = new std::vector<float>;
  m_taugen_etvis = new std::vector<float>;
  m_taugen_z = new std::vector<float>;
  m_taugen_class = new std::vector<int>;


  eventTree = fs->make<TTree>("eventTree","Event Tree");

  eventTree->Branch("tau_pt",&m_tau_pt);
  eventTree->Branch("tau_eta",&m_tau_eta);
  eventTree->Branch("tau_phi",&m_tau_phi);
  eventTree->Branch("tau_z",&m_tau_z);

  eventTree->Branch("taugen_pt",&m_taugen_pt);
  eventTree->Branch("taugen_eta",&m_taugen_eta);
  eventTree->Branch("taugen_phi",&m_taugen_phi);
  eventTree->Branch("taugen_etvis",&m_taugen_etvis);
  eventTree->Branch("taugen_z",&m_taugen_z);
  eventTree->Branch("taugen_class",&m_taugen_class);


}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TrackTauAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
L1TrackTauAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
L1TrackTauAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
L1TrackTauAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
L1TrackTauAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TrackTauAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackTauAnalyzer);
