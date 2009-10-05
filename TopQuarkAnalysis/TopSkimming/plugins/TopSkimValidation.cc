// -*- C++ -*-
//
// Package:    TopSkimValidation
// Class:      TopSkimValidation
// 
/**\class TopSkimValidation TopSkimValidation.cc TopQuarkAnalysis/TopSkimValidation/src/TopSkimValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Puneeth Kalavase
//         Created:  Thu Oct  1 17:22:48 PDT 2009
// $Id$
//
//


#include "TopSkimValidation.h"
#include "TString.h"
// user include files

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaR.h"

typedef math::XYZTLorentzVectorD LorentzVector;

TopSkimValidation::TopSkimValidation(const edm::ParameterSet& iConfig) {

  processName_   = iConfig.getParameter<std::string>("processName");
  triggerNames_  = iConfig.getUntrackedParameter<std::vector<std::string> >("triggerNames");
  leptonPtcut_   = iConfig.getParameter<double>("leptonPtcut");
  HLTPtcut_      = iConfig.getParameter<double>("HLTPtcut"); 
  leptonFlavor_  = iConfig.getParameter<std::string>("leptonFlavor");
  leptonInputTag_= iConfig.getParameter<edm::InputTag>("leptonInputTag"); //input tag of the collection 

}


TopSkimValidation::~TopSkimValidation() {

}


// ------------ method called to for each event  ------------
void
TopSkimValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  using namespace reco;
  
  //make a muon handle in case we're getting muons
  Handle<View<Muon> > mus_h;
  
  //make an electron handle in case we're looking at electrons
  Handle<View<GsfElectron> > els_h;

  //make a vector of lepton p4s
  vector<LorentzVector> lepp4s;

  //make a of p4s with the HLT objects
  vector<LorentzVector> HLTp4s;
  
  if(leptonFlavor_ == "els") {
    iEvent.getByLabel(leptonInputTag_, els_h);
    for(edm::View<GsfElectron>::const_iterator elIt = els_h->begin();
	elIt != els_h->end(); elIt++) {

      if(elIt->pt() < leptonPtcut_)
	continue;
      
      lepp4s.push_back(elIt->p4());
    }
  }


  if(leptonFlavor_ == "mus") {
    iEvent.getByLabel(leptonInputTag_, mus_h);
    for(edm::View<Muon>::const_iterator muIt = mus_h->begin();
	muIt != mus_h->end(); muIt++) {

      //if not a global muon, fail
      if(!(muIt->isGlobalMuon()))
	continue;
      
      if(muIt->pt() < leptonPtcut_)
	continue;
      
      lepp4s.push_back(muIt->p4());
    }
  }

  
  //get the Trigger stuff
  iEvent.getByLabel(edm::InputTag("TriggerResults", "", processName_), triggerResultsH_);
   if (! triggerResultsH_.isValid())
     throw cms::Exception("HLTMaker::produce: error getting TriggerResults product from Event!");
   iEvent.getByLabel(edm::InputTag("hltTriggerSummaryAOD", "", processName_), triggerEventH_  );
   if (! triggerEventH_.isValid()  )
     throw cms::Exception("HLTMaker::produce: error getting TriggerEvent product from Event!"  );
      
    // sanity check
    assert(triggerResultsH_->size()==hltConfig_.size());
    unsigned int nTriggers = triggerResultsH_->size();
        
    for(unsigned int i = 0; i < nTriggers; i++) {
      
      // What is your name?
      const string& name = hltConfig_.triggerName(i);
      if(find(triggerNames_.begin(), triggerNames_.end(), name) == triggerNames_.end())
	continue;

      //get the trigger p4s
      fillTriggerObjectInfo(i, HLTp4s, HLTPtcut_);

    }
    
    //count the number of leptons with pt > 20 GeV
    numPt = numPt + lepp4s.size();

    //how many reco leptons with pt > 20 are matched to a HLT object?
    for(unsigned int j = 0; j < lepp4s.size(); j++) {
      double mindR = 0.15;
      for(unsigned int i = 0; i < HLTp4s.size(); i++) {
	double tempdR = deltaR(HLTp4s.at(i), lepp4s.at(j));
	if(tempdR < mindR ) 
	  mindR = tempdR;
      }//HLT p4 loop
      if(mindR < 0.15) {
	numPtHLTMatched++;
	continue;
      }
    }//lepton p4 loop
    
}


// ------------ method called once each job just before starting event loop  ------------
void TopSkimValidation::beginJob() {

  // HLT config does not change within runs!
  if (hltConfig_.init(processName_)) {
  } else 
    throw cms::Exception("HLTMaker::beginRun: config extraction failure with process name " + processName_);
  
  
  numPt = 0;
  numPtHLTMatched = 0;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
TopSkimValidation::endJob() {
  
  std::cout  << __FILE__ << " " << numPt << "  " << numPtHLTMatched << std::endl;
  
}

void TopSkimValidation::fillTriggerObjectInfo(unsigned int triggerIndex,
					      std::vector<LorentzVector>& p4V,
					      double ptcut) const
{

  
  
  using namespace std;
    const trigger::TriggerObjectCollection& triggerObjects = triggerEventH_->getObjects();
    if (triggerObjects.size() == 0) return;

    // modules on this trigger path
    const vector<string>& moduleLabels = hltConfig_.moduleLabels(triggerIndex);
    // index (slot position) of module giving the decision of the path
    const unsigned int moduleIndex = triggerResultsH_->index(triggerIndex);

    unsigned int nFilters = triggerEventH_->sizeFilters();
    // the first and last filter information is stored
    // but we just want the last filter
    unsigned int lastFilterIndex = nFilters;
    for(unsigned int j = 0; j <= moduleIndex; ++j) {
        const string& moduleLabel = moduleLabels[j];
        const unsigned int filterIndex = triggerEventH_->filterIndex(edm::InputTag(moduleLabel, "", processName_));
        if (filterIndex < nFilters)
            lastFilterIndex = filterIndex;
    }
    if (lastFilterIndex < nFilters) {
        const trigger::Vids& triggerIds = triggerEventH_->filterIds(lastFilterIndex);
        const trigger::Keys& triggerKeys = triggerEventH_->filterKeys(lastFilterIndex);
        assert(triggerIds.size()==triggerKeys.size());

        for(unsigned int j = 0; j < triggerKeys.size(); ++j) {
            const trigger::TriggerObject& triggerObject = triggerObjects[triggerKeys[j]];
	    if(triggerObject.particle().pt() > ptcut)
	      p4V.push_back( LorentzVector( triggerObject.particle().p4() ) );
	}
    }
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(TopSkimValidation);
