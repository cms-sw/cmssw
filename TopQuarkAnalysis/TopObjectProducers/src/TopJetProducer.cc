//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.cc,v 1.3 2007/06/10 08:57:28 lowette Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <memory>


//
// constructors and destructor
//

TopJetProducer::TopJetProducer(const edm::ParameterSet& iConfig) {
  // initialize the configurables
  jetTagsLabel_            = iConfig.getParameter<std::string>   ("jetTagInput");
  caliJetResoFile_         = iConfig.getParameter<std::string>   ("caliJetResoFile");
  recJetsLabel_            = iConfig.getParameter<edm::InputTag> ("recJetInput");
  caliJetsLabel_           = iConfig.getParameter<edm::InputTag> ("caliJetInput");
  addResolutions_             = iConfig.getParameter<bool>       ("addResolutions");
  dropTrackCountingFromAOD    = iConfig.getParameter<bool>       ("dropTrackCountingFromAOD");
  dropTrackProbabilityFromAOD = iConfig.getParameter<bool>       ("dropTrackProbabilityFromAOD");
  dropSoftMuonFromAOD         = iConfig.getParameter<bool>       ("dropSoftMuonFromAOD");
  dropSoftElectronFromAOD     = iConfig.getParameter<bool>       ("dropSoftElectronFromAOD");
  keepdiscriminators          = iConfig.getParameter<bool>       ("keepdiscriminators");
  keepjettagref               = iConfig.getParameter<bool>       ("keepjettagref");
  
  //for jet flavour
  jfi = JetFlavourIdentifier(iConfig.getParameter<edm::ParameterSet>("jetIdParameters"));

  
  // construct resolution calculator
  if (addResolutions_) theResoCalc_ = new TopObjectResolutionCalc(caliJetResoFile_);

  // produces vector of jets
  produces<std::vector<TopJet> >();
}


TopJetProducer::~TopJetProducer() {
  if(addResolutions_) delete theResoCalc_;
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



  vector< Handle< vector<JetTag> > > jetTags_testManyByType ;
  iEvent.getManyByType(jetTags_testManyByType); 
   

  edm::Handle<SoftLeptonTagInfoCollection> jetsInfoHandle_sl;
  edm::Handle<TrackProbabilityTagInfoCollection> jetsInfoHandleTP;
  edm::Handle<TrackCountingTagInfoCollection> jetsInfoHandleTC;
  
  //for jet flavour
  jfi.readEvent(iEvent);


  // loop over jets
  std::vector<TopJet> * topJets = new std::vector<TopJet>(); 
  for (size_t j = 0; j < recjets->size(); j++) {
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
	JetFlavour jetFlavour = jfi.identifyBasedOnPartons((*recjets)[j]);
	int  flavour = jetFlavour.flavour();
	ajet.setQuarkFlavour(flavour);
      }
    }
    // if cal jet found...
    if (cjFound) {
      // add b-tag info if available
      for(size_t k=0; k<jetTags_testManyByType.size(); k++){
	Handle< vector<JetTag> > jetTags = jetTags_testManyByType[k];
	
	
	//**************************
	//get label and module names
	string moduleTagInfoName = (jetTags).provenance()->moduleName();	 
	string moduleLabel = (jetTags).provenance()->moduleLabel();
	//********drop taggers from AOD*********
	if(  (moduleLabel == "trackCountingJetTags"    && dropTrackCountingFromAOD    == true ) ) continue;
	if(  (moduleLabel == "trackProbabilityJetTags" && dropTrackProbabilityFromAOD == true ) ) continue;
	if(  (moduleLabel == "softMuonJetTags"         && dropSoftMuonFromAOD         == true ) ) continue;
	if(  (moduleLabel == "softElectronJetTags    " && dropSoftElectronFromAOD     == true ) ) continue;
	
	for (size_t t = 0; t < jetTags->size(); t++) {
	  // FIXME: is this 0.0001 matching fullproof?
	  
	  
	  // cout << "jet test " << ajet.getLCalJet().et() << "   " << (*jetTags)[t].jet().et()  << endl;
	  //cout << "deltaR   " <<  DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) << endl;
	  if (DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) < 0.00001) {
	    if(jetTagsLabel_ == moduleLabel) ajet.setBdiscriminant((*jetTags)[t].discriminator());
	    
	    
	    
	    //FIXME add combined tagger
	    //********store discriminators*********
	    if(keepdiscriminators == true){
	      pair<string, double> pairdiscri;
	      pairdiscri.first = moduleLabel;
	      pairdiscri.second = (*jetTags)[t].discriminator();
	      //drop TauTag!!!
	      if(moduleTagInfoName == "TrackProbability" || moduleTagInfoName == "TrackCounting" || moduleTagInfoName == "SoftLepton" ){
		ajet.AddBdiscriminantPair(pairdiscri);
	      }
	    }
	    
	    //FIXME add combined tagger
	    //********store jetTagRef*********
	    if(keepjettagref == true){
	      
	      pair<string, JetTagRef> pairjettagref;
	      pairjettagref.first = moduleLabel;
	      
	      if(moduleTagInfoName == "TrackProbability"){
		//cout << "string module label " << moduleLabel << endl;
		iEvent.getByLabel(moduleLabel,jetsInfoHandleTP);  
		const  TrackProbabilityTagInfoCollection & tagInfo_prob = *(jetsInfoHandleTP);
		pairjettagref.second = tagInfo_prob[t].getJetTag();
		ajet.AddBJetTagRefPair(pairjettagref);
	      }
	      if(moduleTagInfoName == "TrackCounting"){
		//cout << "string module label " << moduleLabel << endl;
		iEvent.getByLabel(moduleLabel,jetsInfoHandleTC);  
		const  TrackCountingTagInfoCollection & tagInfo_prob = *(jetsInfoHandleTC);
		pairjettagref.second = tagInfo_prob[t].getJetTag();
		ajet.AddBJetTagRefPair(pairjettagref);
	      }
	      if(moduleTagInfoName == "SoftLepton"){
		// cout << "string module label " << moduleLabel << endl;
		iEvent.getByLabel(moduleLabel,jetsInfoHandle_sl);  
		const  SoftLeptonTagInfoCollection & tagInfo_prob = *(jetsInfoHandle_sl);
		pairjettagref.second = tagInfo_prob[t].getJetTag();
		ajet.AddBJetTagRefPair(pairjettagref);
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
