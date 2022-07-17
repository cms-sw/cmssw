// -*- C++ -*-
//
// Package:    TopQuarkAnalysis/TopTools
// Class:      TestGenTtbarCategories
//
/**\class TestGenTtbarCategories TestGenTtbarCategories.cc PhysicsTools/JetMCAlgos/test/TestGenTtbarCategories.cc

 Description: Analyzer for testing corresponding producer GenTtbarCategorizer

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Johannes Hauk, Nazar Bartosik
//         Created:  Sun, 14 Jun 2015 21:00:23 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <TTree.h>

//
// class declaration
//

class TestGenTtbarCategories : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TestGenTtbarCategories(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------

  // Input tags
  const edm::EDGetTokenT<int> genTtbarIdToken_;

  // Variables to fill
  int ttbarId_;
  int ttbarAdditionalJetId_;
  int nBjetsFromTop_;
  int nBjetsFromW_;
  int nCjetsFromW_;

  // Tree to be filled
  TTree* tree_;
};

//
// constructors and destructor
//
TestGenTtbarCategories::TestGenTtbarCategories(const edm::ParameterSet& iConfig)
    : genTtbarIdToken_(consumes<int>(iConfig.getParameter<edm::InputTag>("genTtbarId"))) {
  usesResource("TFileService");
}

//
// member functions
//

// ------------ method called for each event  ------------
void TestGenTtbarCategories::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<int> genTtbarId;
  iEvent.getByToken(genTtbarIdToken_, genTtbarId);

  // ID including information about b/c jets in acceptance from t->b/W->b/W->c decays as well as additional ones
  ttbarId_ = *genTtbarId;

  // ID based only on additional b/c jets
  ttbarAdditionalJetId_ = ttbarId_ % 100;

  // Number of b/c jets from t->b or W->b/c decays
  nBjetsFromTop_ = ttbarId_ % 1000 / 100;
  nBjetsFromW_ = ttbarId_ % 10000 / 1000;
  nCjetsFromW_ = ttbarId_ % 100000 / 10000;

  // Filling the tree
  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void TestGenTtbarCategories::beginJob() {
  edm::Service<TFileService> fileService;
  if (!fileService)
    throw edm::Exception(edm::errors::Configuration, "TFileService is not registered in cfg file");

  tree_ = fileService->make<TTree>("tree", "tree");
  tree_->Branch("ttbarId", &ttbarId_);
  tree_->Branch("ttbarAdditionalJetId", &ttbarAdditionalJetId_);
  tree_->Branch("nBjetsFromTop", &nBjetsFromTop_);
  tree_->Branch("nBjetsFromW", &nBjetsFromW_);
  tree_->Branch("nCjetsFromW", &nCjetsFromW_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TestGenTtbarCategories::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genTtbarId");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGenTtbarCategories);
