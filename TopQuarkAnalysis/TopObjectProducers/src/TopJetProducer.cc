//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.cc,v 1.7 2007/06/15 16:49:19 heyninck Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonTrackerIsolationPt.h"

#include <vector>
#include <memory>


//
// constructors and destructor
//

TopJetProducer::TopJetProducer(const edm::ParameterSet& iConfig) {
  // initialize the configurables
  recJetsLabel_            	= iConfig.getParameter<edm::InputTag> 	("recJetInput");
  caliJetsLabel_           	= iConfig.getParameter<edm::InputTag> 	("caliJetInput");
  jetTagsLabel_            	= iConfig.getParameter<edm::InputTag>  	("jetTagInput");
  //topElectronsLabel_       	= iConfig.getParameter<edm::InputTag> 	("topElectronsInput");
  //topMuonsLabel_           	= iConfig.getParameter<edm::InputTag> 	("topMuonsInput");
  //doJetCleaning_           	= iConfig.getParameter<bool> 		("doJetCleaning");
  addResolutions_             	= iConfig.getParameter<bool>       	("addResolutions");
  storeBDiscriminants_         	= iConfig.getParameter<bool>       	("storeBDiscriminants");
  dropTrackCountingFromAOD_    	= iConfig.getParameter<bool>       	("dropTrackCountingFromAOD");
  dropTrackProbabilityFromAOD_ 	= iConfig.getParameter<bool>       	("dropTrackProbabilityFromAOD");
  dropSoftMuonFromAOD_         	= iConfig.getParameter<bool>       	("dropSoftMuonFromAOD");
  dropSoftElectronFromAOD_     	= iConfig.getParameter<bool>       	("dropSoftElectronFromAOD");
  keepdiscriminators_          	= iConfig.getParameter<bool>       	("keepdiscriminators");
  keepjettagref_               	= iConfig.getParameter<bool>       	("keepjettagref");
  caliJetResoFile_         	= iConfig.getParameter<std::string>   	("caliJetResoFile");

  //LEPJETDR_=0.3;//deltaR cut used to associate a jet to an electron for jet cleaning.  Make it configurable?
  //ELEISOCUT_=0.1;//cut on electron isolation for jet cleaning
  //MUISOCUT_=0.1;//cut on muon isolation for jet cleaning
    
  // construct the jet flavour identifier
  jetFlavId_ =  new JetFlavourIdentifier(iConfig.getParameter<edm::ParameterSet>("jetIdParameters"));
  
  // construct resolution calculator
  if (addResolutions_) theResoCalc_ = new TopObjectResolutionCalc(caliJetResoFile_);

  // produces vector of jets
  produces<std::vector<TopJet> >();
}


TopJetProducer::~TopJetProducer() {
  if(addResolutions_) delete theResoCalc_;
  delete jetFlavId_;
}


//
// member functions
//

void TopJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of generated particles from the event
  edm::Handle<std::vector<JetType> > recjets;
  iEvent.getByLabel(recJetsLabel_, recjets);
  edm::Handle<std::vector<JetType> > calijets;
  iEvent.getByLabel(caliJetsLabel_, calijets);
  //edm::Handle<std::vector<TopElectron> > electronsHandle;
  //iEvent.getByLabel(topElectronsLabel_, electronsHandle);
  //std::vector<TopElectron> electrons=*electronsHandle;
  //edm::Handle<std::vector<TopMuon> > muonsHandle;
  //iEvent.getByLabel(topMuonsLabel_, muonsHandle);
  //std::vector<TopMuon> muons=*muonsHandle;



  std::vector<edm::Handle<std::vector<reco::JetTag> > > jetTags_testManyByType ;
  iEvent.getManyByType(jetTags_testManyByType); 

  edm::Handle<reco::SoftLeptonTagInfoCollection> jetsInfoHandle_sl;
  edm::Handle<reco::TrackProbabilityTagInfoCollection> jetsInfoHandleTP;
  edm::Handle<reco::TrackCountingTagInfoCollection> jetsInfoHandleTC;
  
  //for jet flavour
  jetFlavId_->readEvent(iEvent);

  //select isolated leptons to remove from jets collection
  //electrons=selectIsolated(electrons,ELEISOCUT_,iSetup,iEvent);
  //muons=selectIsolated(muons,MUISOCUT_,iSetup,iEvent);


  // loop over jets
  std::vector<TopJet> * topJets = new std::vector<TopJet>(); 
  for (size_t j = 0; j < recjets->size(); j++) {
/* 
    //check that the jet doesn't match in deltaR with an isolated lepton
    //if it does, then it needs to be cleaned (ie don't put it in the TopJet collection)
    //FIXME: don't do muons until have a sensible cut value on their isolation
    float mindr=9999.;
    for (size_t ie=0; ie<electrons.size(); ie++) {
      float dr=DeltaR<reco::Candidate>()((*recjets)[j],electrons[ie]);
      if (dr<mindr) {
	mindr=dr;
      }
    }
    //if the jet is closely matched in dR to electron, skip it
    if (mindr<LEPJETDR_ && doJetCleaning_) {
      continue;
    }

*/
    // construct the TopJet
    TopJet ajet;
    // loop over cal jets to find corresponding jet
    bool cjFound = false;
    for (size_t cj = 0; cj < calijets->size(); cj++) {
      // FIXME: is this 0.01 matching fullproof?
      if (DeltaR<reco::Candidate>()((*recjets)[j], (*calijets)[cj]) < 0.01) {
        cjFound = true;
        ajet = TopJet((*calijets)[cj]);
        ajet.setRecJet((*recjets)[j]);
	JetFlavour jetFlavour = jetFlavId_->identifyBasedOnPartons((*recjets)[j]);
	int  flavour = jetFlavour.flavour();
	ajet.setQuarkFlavour(flavour);
      }
    }
    // if cal jet found...
    if (cjFound) {
      // add b-tag info if available & required
      if(storeBDiscriminants_){
        for(size_t k=0; k<jetTags_testManyByType.size(); k++){
	  edm::Handle<std::vector<reco::JetTag> > jetTags = jetTags_testManyByType[k];


	  //**************************
	  //get label and module names
	  std::string moduleTagInfoName = (jetTags).provenance()->moduleName();	 
	  std::string moduleLabel = (jetTags).provenance()->moduleLabel();
	  //********drop taggers from AOD*********
	  if(  (moduleLabel == "trackCountingJetTags"    && dropTrackCountingFromAOD_    == true ) ) continue;
	  if(  (moduleLabel == "trackProbabilityJetTags" && dropTrackProbabilityFromAOD_ == true ) ) continue;
	  if(  (moduleLabel == "softMuonJetTags"         && dropSoftMuonFromAOD_         == true ) ) continue;
	  if(  (moduleLabel == "softElectronJetTags    " && dropSoftElectronFromAOD_     == true ) ) continue;
	
	  for (size_t t = 0; t < jetTags->size(); t++) {
	    // FIXME: is this 0.0001 matching fullproof?
	  
	  
	    // cout << "jet test " << ajet.getLCalJet().et() << "   " << (*jetTags)[t].jet().et()  << endl;
	    //cout << "deltaR   " <<  DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) << endl;
	    if (DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) < 0.00001) {
	      if(jetTagsLabel_.label() == moduleLabel) ajet.setBdiscriminant((*jetTags)[t].discriminator());
	    
	    
	    
	      //FIXME add combined tagger
	      //********store discriminators*********
	      if(keepdiscriminators_ == true){
	        std::pair<std::string, double> pairdiscri;
	        pairdiscri.first = moduleLabel;
	        pairdiscri.second = (*jetTags)[t].discriminator();
	        //drop TauTag!!!
	        if(moduleTagInfoName == "TrackProbability" || moduleTagInfoName == "TrackCounting" || moduleTagInfoName == "SoftLepton" ){
		  ajet.addBdiscriminantPair(pairdiscri);
	        }
	      }
	    
	      //FIXME add combined tagger
	      //********store jetTagRef*********
	      if(keepjettagref_ == true){
	      
	        std::pair<std::string, reco::JetTagRef> pairjettagref;
	        pairjettagref.first = moduleLabel;
	      
	        if(moduleTagInfoName == "TrackProbability"){
		  //cout << "string module label " << moduleLabel << endl;
		  iEvent.getByLabel(moduleLabel,jetsInfoHandleTP);  
		  const  reco::TrackProbabilityTagInfoCollection & tagInfo_prob = *(jetsInfoHandleTP);
		  pairjettagref.second = tagInfo_prob[t].getJetTag();
		  ajet.addBJetTagRefPair(pairjettagref);
	        }
	        if(moduleTagInfoName == "TrackCounting"){
		  //cout << "string module label " << moduleLabel << endl;
		  iEvent.getByLabel(moduleLabel,jetsInfoHandleTC);  
		  const  reco::TrackCountingTagInfoCollection & tagInfo_prob = *(jetsInfoHandleTC);
		  pairjettagref.second = tagInfo_prob[t].getJetTag();
		  ajet.addBJetTagRefPair(pairjettagref);
	        }  
	        if(moduleTagInfoName == "SoftLepton"){
		  // cout << "string module label " << moduleLabel << endl;
		  iEvent.getByLabel(moduleLabel,jetsInfoHandle_sl);  
		  const  reco::SoftLeptonTagInfoCollection & tagInfo_prob = *(jetsInfoHandle_sl);
		  pairjettagref.second = tagInfo_prob[t].getJetTag();
		  ajet.addBJetTagRefPair(pairjettagref);
	        }
	      }  
	    }
	  }
	}
      }
      // add resolution info if demanded
      if (addResolutions_) {
        (*theResoCalc_)(ajet);
      }
    } else {
      std::cout << "no cal jet found " << std::endl;
    }
    topJets->push_back(ajet);
  }

  // sort jets in ET
  std::sort(topJets->begin(), topJets->end(), eTComparator_);

  // put genEvt  in Event
  std::auto_ptr<std::vector<TopJet> > myTopJetProducer(topJets);
  iEvent.put(myTopJetProducer);

}
/*

//takes a vector of electrons and returns a vector that only contains the ones that are isolated
//isolation is calculated by TopLeptonTrackerIsolationPt.  The second argument is the isolation cut
//to use
std::vector<TopElectron> TopJetProducer::selectIsolated(const std::vector<TopElectron> &electrons, float isoCut,
							const edm::EventSetup &iSetup, const edm::Event &iEvent) {
  

  TopLeptonTrackerIsolationPt tkIsoPtCalc(iSetup);
  std::vector<TopElectron> output;
  for (size_t ie=0; ie<electrons.size(); ie++) {
    
    if (tkIsoPtCalc.calculate(electrons[ie],iEvent) < isoCut) {
      output.push_back(electrons[ie]);
    }

  }
  
  return output;
}

//takes a vector of muons and returns a vector that only contains the ones that are isolated
//isolation is calculated by TopLeptonTrackerIsolationPt.  The second argument is the isolation cut
//to use
//FIXME I could combine this with the one for electrons using templates?
std::vector<TopMuon> TopJetProducer::selectIsolated(const std::vector<TopMuon> &muons, float isoCut,
							const edm::EventSetup &iSetup, const edm::Event &iEvent) {
  

  TopLeptonTrackerIsolationPt tkIsoPtCalc(iSetup);
  std::vector<TopMuon> output;
  for (size_t iu=0; iu<muons.size(); iu++) {
    
    if (tkIsoPtCalc.calculate(muons[iu],iEvent) < isoCut) {
      output.push_back(muons[iu]);
    }
    
  }
  
  return output;
}
*/
