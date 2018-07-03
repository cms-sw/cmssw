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
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class TestGenTtbarCategories : public edm::EDAnalyzer {
    public:
        explicit TestGenTtbarCategories(const edm::ParameterSet&);
        ~TestGenTtbarCategories();
        
        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
        
        
    private:
        virtual void beginJob() override;
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
        virtual void endJob() override;
        
        //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
        //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
        //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        
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
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestGenTtbarCategories::TestGenTtbarCategories(const edm::ParameterSet& iConfig):
genTtbarIdToken_(consumes<int>(iConfig.getParameter<edm::InputTag>("genTtbarId")))
{}


TestGenTtbarCategories::~TestGenTtbarCategories()
{}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestGenTtbarCategories::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<int> genTtbarId;
    iEvent.getByToken(genTtbarIdToken_, genTtbarId);
    
    // ID including information about b/c jets in acceptance from t->b/W->b/W->c decays as well as additional ones
    ttbarId_ = *genTtbarId;
    
    // ID based only on additional b/c jets
    ttbarAdditionalJetId_ = ttbarId_%100;
    
    // Number of b/c jets from t->b or W->b/c decays
    nBjetsFromTop_ = ttbarId_%1000/100;
    nBjetsFromW_ = ttbarId_%10000/1000;
    nCjetsFromW_ = ttbarId_%100000/10000;
    
    // Filling the tree
    tree_->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestGenTtbarCategories::beginJob()
{
    edm::Service<TFileService> fileService;
    if(!fileService) throw edm::Exception(edm::errors::Configuration, "TFileService is not registered in cfg file");
    
    tree_ = fileService->make<TTree>("tree", "tree");
    tree_->Branch("ttbarId", &ttbarId_);
    tree_->Branch("ttbarAdditionalJetId", &ttbarAdditionalJetId_);
    tree_->Branch("nBjetsFromTop", &nBjetsFromTop_);
    tree_->Branch("nBjetsFromW", &nBjetsFromW_);
    tree_->Branch("nCjetsFromW", &nCjetsFromW_);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestGenTtbarCategories::endJob() 
{}

// ------------ method called when starting to processes a run  ------------
/*
void 
TestGenTtbarCategories::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
TestGenTtbarCategories::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
TestGenTtbarCategories::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
TestGenTtbarCategories::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TestGenTtbarCategories::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGenTtbarCategories);
