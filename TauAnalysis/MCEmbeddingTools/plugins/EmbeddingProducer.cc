// -*- C++ -*-
//
// Package:    TauAnalysis_new/EmbeddingProducer
// Class:      EmbeddingProducer
// 
/**\class EmbeddingProducer EmbeddingProducer.cc TauAnalysis_new/EmbeddingProducer/plugins/EmbeddingProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stefan Wayand
//         Created:  Wed, 09 Dec 2015 13:14:54 GMT
//
//


// system include files
#include <memory>

// user include files
#include "TH1F.h"
#include "TFile.h"
#include "TString.h"
#include "TVector2.h" 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "CommonTools/CandAlgos/interface/CandCombiner.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "GeneratorInterface/Pythia8Interface/interface/Py8InterfaceBase.h"

//
// class declaration
//

class EmbeddingProducer : public edm::EDProducer {
   public:
      explicit EmbeddingProducer(const edm::ParameterSet&);
      ~EmbeddingProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      void reset_event_content();
      void add_to_event(edm::Event& iEvent);
      void match_count_and_fill(TString, std::vector<pat::Muon>::const_iterator);
      void count_and_fill(TString, TString, std::vector<pat::Muon>::const_iterator);
      
      // ----------member data ---------------------------
      edm::EDGetTokenT<pat::MuonCollection> muonsCollection_;
      edm::EDGetTokenT<reco::VertexCollection> vtxCollection_;
      edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
      edm::EDGetTokenT<pat::MuonCollection> patMuonsAfterKinCuts_;
      edm::EDGetTokenT<reco::CompositeCandidateCollection> ZmumuCandidates_;
      edm::InputTag srcHepMC_;
      bool mixHepMC_;


      // the "fake" MC event from the embedded source
      std::unique_ptr<HepMC::GenEvent> genEvent_;
      std::unique_ptr<GenEventInfoProduct> genEventInfo_;
      
      
      // How often does the embedded event pass the kinematic requirments (pt and eta)
      unsigned int numEvents_tried;
      unsigned int numEvents_passed;
      
      // Histograms and root output files
      TFile* histFile;
      TString histFileName;
      std::vector<TString> selection = {"genfilter","baseline","id","id_and_trigger"};
      std::vector<TString> matchingMC = {"all","MC_matched","not_MC_matched"};
      std::vector<TString> string_keys;
      std::map<TString,TH1F*> nMuons;
      std::map<TString,int> nMuonsNumbers;
      std::map<TString,TH1F*> ptMuons;
      std::map<TString,TH1F*> etaMuons; 
};

//
// constructors and destructor
//
EmbeddingProducer::EmbeddingProducer(const edm::ParameterSet& iConfig){   
  
  histFileName = TString(iConfig.getParameter<std::string>("histFileName"));
  histFile = new TFile(histFileName,"RECREATE");
  for (unsigned int i=0;i<selection.size();++i)
  {
    TDirectory* selection_dir = histFile->mkdir(selection[i]);
    for (unsigned int j=0;j<matchingMC.size();++j)
    {
      TDirectory* full_dir = selection_dir->mkdir(matchingMC[j]);
      TString string_key = selection[i]+TString("_")+matchingMC[j];
      string_keys.push_back(string_key);
      
      Double_t n[5] = {0.,2.,3.,4.,5.};
      nMuons[string_key] = new TH1F("nMuons","nMuons",4,n);
      nMuons[string_key]->SetDirectory(full_dir);
      nMuonsNumbers[string_key] = 0;
      
      ptMuons[string_key] = new TH1F("ptMuons","ptMuons",50,0,150);
      ptMuons[string_key]->SetDirectory(full_dir);
      
      etaMuons[string_key] = new TH1F("etaMuons","etaMuons",50,-3,3);
      etaMuons[string_key]->SetDirectory(full_dir);
    }
  }
  
  muonsCollection_ = consumes<pat::MuonCollection>(iConfig.getParameter<edm::InputTag>("src"));
  vtxCollection_ = consumes<reco::VertexCollection>(iConfig.getParameter< edm::InputTag >("vtxSrc"));
  triggerResults_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","","HLT"));
  patMuonsAfterKinCuts_ = consumes<pat::MuonCollection>(edm::InputTag("patMuonsAfterKinCuts","","EMBS"));
  ZmumuCandidates_ = consumes<reco::CompositeCandidateCollection>(edm::InputTag("ZmumuCandidates","","EMBS"));
  mixHepMC_ = iConfig.getParameter<bool>("mixHepMc");
  if (mixHepMC_) srcHepMC_ = iConfig.getParameter<edm::InputTag>("hepMcSrc");
  
  produces<GenFilterInfo>("minVisPtFilter");
  produces<GenEventInfoProduct>("");
}


EmbeddingProducer::~EmbeddingProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EmbeddingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  reset_event_content();
  
  using namespace edm;
  
  // reading out baseline selection results with corresponding collections
  Handle<pat::MuonCollection> patMuonsAfterKinCuts;
  iEvent.getByToken(patMuonsAfterKinCuts_, patMuonsAfterKinCuts);
  bool passed_kin_cuts = (patMuonsAfterKinCuts->size() > 0);
  
  Handle<reco::CompositeCandidateCollection> ZmumuCandidates;
  iEvent.getByToken(ZmumuCandidates_,ZmumuCandidates);
  bool is_z_candidate = (ZmumuCandidates->size() > 0);
  
  // reading out necessary collections for analysis
  Handle<std::vector<pat::Muon>> coll_muons;
  iEvent.getByToken(muonsCollection_ , coll_muons);
  
  Handle<reco::VertexCollection> offlinePrimaryVertices;
  iEvent.getByToken(vtxCollection_,offlinePrimaryVertices);
  
  // reading out trigger results
  Handle<TriggerResults> trigResults;
  iEvent.getByToken(triggerResults_,trigResults);
  const TriggerNames& trigNames = iEvent.triggerNames(*trigResults);   
  std::string pathName1 = "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v1";
  std::string pathName2 = "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v1";
  bool passedTrigger1  = trigResults->accept(trigNames.triggerIndex(pathName1));  
  bool passedTrigger2  = trigResults->accept(trigNames.triggerIndex(pathName2));
  unsigned key=0;
  for (std::vector<pat::Muon>::const_iterator muon=  coll_muons->begin(); muon!= coll_muons->end();  ++muon,  ++key){
    
    match_count_and_fill(selection[0],muon); // choosing "genfilter" selection
    
    if (passed_kin_cuts && is_z_candidate)
    {
      match_count_and_fill(selection[1],muon); // choosing "baseline" selection
      //if ( muon->isTightMuon(*offlinePrimaryVertices->begin()) )
      if ( muon->isMediumMuon())
      {
        match_count_and_fill(selection[2],muon); // choosing "id" selection
        if (passedTrigger1 || passedTrigger2)
        {
          match_count_and_fill(selection[3],muon); // choosing "id_and_trigger" selection
        }
      }
    }
  }
  
  for (unsigned int i=0;i<string_keys.size();++i)
  { 
    nMuons[string_keys[i]]->Fill(nMuonsNumbers[string_keys[i]]);
    nMuonsNumbers[string_keys[i]] = 0;
  }
  
  
  add_to_event(iEvent);
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
EmbeddingProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmbeddingProducer::endJob() {
  
  histFile->Write();
  histFile->Close();
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EmbeddingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}



// user functions 
void 
EmbeddingProducer::reset_event_content(){
  
  numEvents_tried  = 0;
  numEvents_passed = 0;
  genEvent_.reset(new HepMC::GenEvent);
  
}

void
EmbeddingProducer::add_to_event(edm::Event& iEvent){
  
    std::unique_ptr<GenFilterInfo> kinfilter(new GenFilterInfo(numEvents_tried, numEvents_passed));
    iEvent.put(std::move(kinfilter), std::string("minVisPtFilter"));
  
  
    std::unique_ptr<GenEventInfoProduct> generator(new GenEventInfoProduct());
    iEvent.put(std::move(generator), std::string(""));
}

void
EmbeddingProducer::match_count_and_fill(TString selection_string, std::vector<pat::Muon>::const_iterator muon)
{
  count_and_fill(selection_string,matchingMC[0],muon); // choosing "all" muons
  bool mc_matched = false;
  
  if(muon->genParticleRefs().size()>0 && muon->genParticle(0) != 0)
  {
    double phi_diff = std::abs(TVector2::Phi_mpi_pi(muon->genParticle(0)->p4().phi() - muon->p4().phi()));
    double eta_diff = std::abs(muon->genParticle(0)->p4().eta() - muon->p4().eta());
    double Delta_R = std::sqrt(phi_diff*phi_diff + eta_diff*eta_diff);
    if (Delta_R < 0.05) mc_matched = true;
  }
  if (mc_matched)
  {
    count_and_fill(selection_string,matchingMC[1],muon); // choosing "MC matched" muons
  }
  else
  {
    count_and_fill(selection_string,matchingMC[2],muon); // choosing "not MC matched" muons
  }
}

void
EmbeddingProducer::count_and_fill(TString selection_string, TString matching_string, std::vector<pat::Muon>::const_iterator muon)
{
  TString string_key = selection_string + TString("_") + matching_string;
  ++nMuonsNumbers[string_key];
  ptMuons[string_key]->Fill(muon->p4().pt());
  etaMuons[string_key]->Fill(muon->p4().eta());
}


//define this as a plug-in
DEFINE_FWK_MODULE(EmbeddingProducer);
