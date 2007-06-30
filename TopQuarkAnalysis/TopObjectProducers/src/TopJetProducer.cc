//
// Author:  Jan Heyninck
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopJetProducer.cc,v 1.11 2007/06/23 07:27:05 lowette Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopJetProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfoFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"
#include "RecoBTag/MCTools/interface/JetFlavourIdentifier.h"
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
  recJetsLabel_                  = iConfig.getParameter<edm::InputTag> ("recJetInput");
  caliJetsLabel_                 = iConfig.getParameter<edm::InputTag> ("caliJetInput");
  // TEMP Jet cleaning from electrons
  topElectronsLabel_           = iConfig.getParameter<edm::InputTag> ("topElectronsInput");
  topMuonsLabel_               = iConfig.getParameter<edm::InputTag> ("topMuonsInput");
  doJetCleaning_               = iConfig.getParameter<bool> 	       ("doJetCleaning");
  // TEMP End
  addResolutions_             	 = iConfig.getParameter<bool>          ("addResolutions");
  caliJetResoFile_               = iConfig.getParameter<std::string>   ("caliJetResoFile");
  storeBTagInfo_                 = iConfig.getParameter<bool>          ("storeBTagInfo");
  ignoreTrackCountingFromAOD_    = iConfig.getParameter<bool>          ("ignoreTrackCountingFromAOD");
  ignoreTrackProbabilityFromAOD_ = iConfig.getParameter<bool>          ("ignoreTrackProbabilityFromAOD");
  ignoreSoftMuonFromAOD_         = iConfig.getParameter<bool>          ("ignoreSoftMuonFromAOD");
  ignoreSoftElectronFromAOD_     = iConfig.getParameter<bool>          ("ignoreSoftElectronFromAOD");
  keepDiscriminators_            = iConfig.getParameter<bool>          ("keepDiscriminators");
  keepJetTagRefs_                = iConfig.getParameter<bool>          ("keepJetTagRefs");
  getJetMCFlavour_               = iConfig.getParameter<bool>          ("getJetMCFlavour");
  computeJetCharge_              = iConfig.getParameter<bool>          ("computeJetCharge"); 
  storeAssociatedTracks_         = iConfig.getParameter<bool>          ("storeAssociatedTracks"); 

  // TEMP Jet cleaning from electrons
  LEPJETDR_=0.3;//deltaR cut used to associate a jet to an electron for jet cleaning.  Make it configurable?
  ELEISOCUT_=0.1;//cut on electron isolation for jet cleaning
  MUISOCUT_=0.1;//cut on muon isolation for jet cleaning
  // TEMP End
    
  // construct resolution calculator
  if (addResolutions_) theResoCalc_ = new TopObjectResolutionCalc(caliJetResoFile_);
  // construct the jet flavour identifier
  if (getJetMCFlavour_) jetFlavId_ = new JetFlavourIdentifier(iConfig.getParameter<edm::ParameterSet>("jetIdParameters"));

  // construct Jet Track Associator
  trackAssociationPSet_     = iConfig.getParameter<edm::ParameterSet>("trackAssociation");
  simpleJetTrackAssociator_ = reco::helper::SimpleJetTrackAssociator(trackAssociationPSet_);      

  // construct Jet Charge Computer
  jetChargePSet_        = iConfig.getParameter<edm::ParameterSet>("jetCharge");
  jetCharge_            = JetCharge(jetChargePSet_);
 
  // produces vector of jets
  produces<std::vector<TopJet> >();
}


TopJetProducer::~TopJetProducer() {
  if (addResolutions_) delete theResoCalc_;
  if (getJetMCFlavour_) delete jetFlavId_;
}


//
// member functions
//

void TopJetProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the vector of non-calibrated jets
  edm::Handle<std::vector<TopJetType> > recjets;
  iEvent.getByLabel(recJetsLabel_, recjets);
  // Get the vector of calibrated jets
  edm::Handle<std::vector<TopJetType> > calijets;
  iEvent.getByLabel(caliJetsLabel_, calijets);
  // TEMP Jet cleaning from electrons
  edm::Handle<std::vector<TopElectron> > electronsHandle;
  iEvent.getByLabel(topElectronsLabel_, electronsHandle);
  std::vector<TopElectron> electrons=*electronsHandle;
  edm::Handle<std::vector<TopMuon> > muonsHandle;
  iEvent.getByLabel(topMuonsLabel_, muonsHandle);
  std::vector<TopMuon> muons=*muonsHandle;
  // TEMP End

  // tracks Jet Track Association, by hand in CMSSW_1_3_X
  edm::Handle<reco::TrackCollection> hTracks;
  iEvent.getByLabel(trackAssociationPSet_.getParameter<edm::InputTag>("tracks"), hTracks);

  // Get the vector of jet tags with b-tagging info
  std::vector<edm::Handle<std::vector<reco::JetTag> > > jetTags_testManyByType ;
  iEvent.getManyByType(jetTags_testManyByType); 
  // Define the handles for the specific algorithms
  edm::Handle<reco::SoftLeptonTagInfoCollection> jetsInfoHandle_sl;
  edm::Handle<reco::TrackProbabilityTagInfoCollection> jetsInfoHandleTP;
  edm::Handle<reco::TrackCountingTagInfoCollection> jetsInfoHandleTC;

  // for jet flavour
  if (getJetMCFlavour_) jetFlavId_->readEvent(iEvent);

  // TEMP Jet cleaning from electrons
  //select isolated leptons to remove from jets collection
  electrons=selectIsolated(electrons,ELEISOCUT_,iSetup,iEvent);
  muons=selectIsolated(muons,MUISOCUT_,iSetup,iEvent);
  // TEMP End


  // loop over jets
  std::vector<TopJet> * topJets = new std::vector<TopJet>(); 
  for (size_t j = 0; j < recjets->size(); j++) {
 
  // TEMP Jet cleaning from electrons
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
  // TEMP End


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
      }
    }
    // if cal jet found...
    if (cjFound) {
      // get the MC flavour information for this jet
      if (getJetMCFlavour_) {
        JetFlavour jetFlavour = jetFlavId_->identifyBasedOnPartons((*recjets)[j]);
        ajet.setPartonFlavour(jetFlavour.flavour());
      }
      // add b-tag info if available & required
      if (storeBTagInfo_) {
        for(size_t k=0; k<jetTags_testManyByType.size(); k++){
	  edm::Handle<std::vector<reco::JetTag> > jetTags = jetTags_testManyByType[k];

	  //**************************
	  //get label and module names
	  std::string moduleTagInfoName = (jetTags).provenance()->moduleName();	 
	  std::string moduleLabel = (jetTags).provenance()->moduleLabel();
	  //********ignore taggers from AOD*********
	  if(  (moduleLabel == "trackCountingJetTags"    && ignoreTrackCountingFromAOD_    == true ) ) continue;
	  if(  (moduleLabel == "trackProbabilityJetTags" && ignoreTrackProbabilityFromAOD_ == true ) ) continue;
	  if(  (moduleLabel == "softMuonJetTags"         && ignoreSoftMuonFromAOD_         == true ) ) continue;
	  if(  (moduleLabel == "softElectronJetTags    " && ignoreSoftElectronFromAOD_     == true ) ) continue;
	
	  for (size_t t = 0; t < jetTags->size(); t++) {
	    // cout << "jet test " << ajet.getLCalJet().et() << "   " << (*jetTags)[t].jet().et()  << endl;
	    //cout << "deltaR   " <<  DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) << endl;

	    // FIXME: is this 0.0001 matching fullproof?
	    if (DeltaR<reco::Candidate>()((*recjets)[j], (*jetTags)[t].jet()) < 0.00001) {
	    
	      //FIXME add combined tagger
	      //********store discriminators*********
	      if(keepDiscriminators_ == true){
	        std::pair<std::string, double> pairDiscri;
	        pairDiscri.first = moduleLabel;
	        pairDiscri.second = (*jetTags)[t].discriminator();
	        //drop TauTag!!!
	        if(moduleTagInfoName == "TrackProbability" || moduleTagInfoName == "TrackCounting" || moduleTagInfoName == "SoftLepton" ){
		  ajet.addBDiscriminatorPair(pairDiscri);
	        }
	      }
	    
	      //FIXME add combined tagger
	      //********store jetTagRef*********
	      if(keepJetTagRefs_ == true){
	      
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

    // Associate tracks with jet (at least temporary)
    simpleJetTrackAssociator_.associate(ajet.momentum(), hTracks, ajet.associatedTracks_);

    // PUT HERE EVERYTHING WHICH NEEDS TRACKS
    if (computeJetCharge_) {
        ajet.jetCharge_ = static_cast<float>(jetCharge_.charge(ajet.p4(), ajet.associatedTracks_));
    }

    // drop jet track association if the user does not want it
    if (!storeAssociatedTracks_) { ajet.associatedTracks_.clear(); }

    // end of TopObjectProducer loop
    topJets->push_back(ajet);
  }

  // sort jets in ET
  std::sort(topJets->begin(), topJets->end(), eTComparator_);

  // put genEvt  in Event
  std::auto_ptr<std::vector<TopJet> > myTopJetProducer(topJets);
  iEvent.put(myTopJetProducer);

}


// TEMP Jet cleaning from electrons
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
// TEMP End


// TEMP Jet cleaning from electrons
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
// TEMP End

