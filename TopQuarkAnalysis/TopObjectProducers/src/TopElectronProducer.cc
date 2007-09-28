//
// $Id: TopElectronProducer.cc,v 1.22 2007/09/21 00:28:15 lowette Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TrackerIsolationPt.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/CaloIsolationEnergy.h"

#include <vector>
#include <memory>


TopElectronProducer::TopElectronProducer(const edm::ParameterSet & iConfig) {

  // general configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>( "electronSource" );
  // ghost removal configurable
  doGhostRemoval_   = iConfig.getParameter<bool>         ( "removeDuplicates" );
  // MC matching configurables
  doGenMatch_       = iConfig.getParameter<bool>         ( "doGenMatch" );
  genPartSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleSource" );
  maxDeltaR_        = iConfig.getParameter<double>       ( "maxDeltaR" );
  minRecoOnGenEt_   = iConfig.getParameter<double>       ( "minRecoOnGenEt" );
  maxRecoOnGenEt_   = iConfig.getParameter<double>       ( "maxRecoOnGenEt" );
  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_        = iConfig.getParameter<bool>         ( "useNNResolution" );
  electronResoFile_ = iConfig.getParameter<std::string>  ( "electronResoFile" );
  // isolation configurables
  doTrkIso_         = iConfig.getParameter<bool>         ( "doTrkIsolation" );
  tracksSrc_        = iConfig.getParameter<edm::InputTag>( "tracksSource" );
  doCalIso_         = iConfig.getParameter<bool>         ( "doCalIsolation" );
  towerSrc_         = iConfig.getParameter<edm::InputTag>( "towerSource" );
  // electron ID configurables
  addElecID_        = iConfig.getParameter<bool>         ( "addElectronID" );
  elecIDSrc_        = iConfig.getParameter<edm::InputTag>( "electronIDSource" );
  // likelihood ratio configurables
  addLRValues_      = iConfig.getParameter<bool>         ( "addLRValues" );
  electronLRFile_   = iConfig.getParameter<std::string>  ( "electronLRFile" );

  // construct resolution calculator
  if(addResolutions_){
    theResoCalc_= new TopObjectResolutionCalc(edm::FileInPath(electronResoFile_).fullPath(), useNNReso_);
  }

  // produces vector of muons
  produces<std::vector<TopElectron> >();

}


TopElectronProducer::~TopElectronProducer() {
  if(addResolutions_) delete theResoCalc_;
}


void TopElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  // Get the collection of muons from the event
  edm::Handle<std::vector<TopElectronType> > electronsHandle;
  iEvent.getByLabel(electronSrc_, electronsHandle);
  std::vector<TopElectronType> electrons = *electronsHandle;

  // remove ghosts
  if (doGhostRemoval_) {
    removeGhosts(electrons);
  }

  // prepare the MC matching
  edm::Handle<reco::CandidateCollection> particles;
  if (doGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
    matchTruth(*particles, electrons) ;
  }

  // prepare isolation calculation
  edm::Handle<reco::TrackCollection> trackHandle;
  if (doTrkIso_) {
    trkIsolation_= new TrackerIsolationPt();
    iEvent.getByLabel(tracksSrc_, trackHandle);
  }
  std::vector<CaloTower> towers;
  if (doCalIso_) {
    calIsolation_= new CaloIsolationEnergy();
    edm::Handle<CaloTowerCollection> towerHandle;
    iEvent.getByLabel(towerSrc_, towerHandle);
    CaloTowerCollection towerColl = *(towerHandle.product());
    for (CaloTowerCollection::const_iterator itTower = towerColl.begin(); itTower != towerColl.end(); itTower++) {
      towers.push_back(*itTower);
    }
  }
  
  // prepare ID extraction
  edm::Handle<reco::ElectronIDAssociationCollection> elecIDs;
  if (addElecID_) iEvent.getByLabel(elecIDSrc_, elecIDs);
  
  // prepare LR calculation
  if(addLRValues_) {
    theLeptonLRCalc_= new TopLeptonLRCalc(iSetup, edm::FileInPath(electronLRFile_).fullPath(), "", "");
  }

  std::vector<TopElectron> * topElectrons = new std::vector<TopElectron>();
  for (size_t e = 0; e < electrons.size(); ++e) {
    // construct the TopElectron
    TopElectron anElectron(electrons[e]);
    // match to generated final state electrons
    if (doGenMatch_) {
      anElectron.setGenLepton(findTruth(*particles, electrons[e]));
    }
    // add resolution info
    if(addResolutions_){
      (*theResoCalc_)(anElectron);
    }
    // do tracker isolation
    if (doTrkIso_) {
      anElectron.setTrackIso(trkIsolation_->calculate(anElectron, *trackHandle));
    }
    // do calorimeter isolation
    if (doCalIso_) {
      anElectron.setCaloIso(calIsolation_->calculate(anElectron, towers));
    }
    // add electron ID info
    if (addElecID_) {
      anElectron.setLeptonID(electronID(electronsHandle, elecIDs, e));
    }
    // add lepton LR info
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(anElectron, iEvent);
    }
    // add sel to selected
    topElectrons->push_back(TopElectron(anElectron));
  }

  // sort electrons in pt
  std::sort(topElectrons->begin(), topElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<TopElectron> > ptr(topElectrons);
  iEvent.put(ptr);

  // clean up
  if (doTrkIso_) delete trkIsolation_;
  if (doCalIso_) delete calIsolation_;
  if (addLRValues_) delete theLeptonLRCalc_;

}


void TopElectronProducer::removeGhosts(std::vector<TopElectronType> & elecs) {
  std::vector<TopElectronType>::iterator cmp = elecs.begin();  
  std::vector<TopElectronType>::iterator ref = elecs.begin();  
  for( ; ref<elecs.end(); ++ref ){
    for( ; (cmp!=ref) && cmp<elecs.end(); ++cmp ){
      if ((cmp->gsfTrack()==ref->gsfTrack()) || (cmp->superCluster()==ref->superCluster()) ){
	//same track or super cluster is used
	//keep the one with E/p closer to one	
	if(fabs(ref->eSuperClusterOverP()-1.) < fabs(cmp->eSuperClusterOverP()-1.)){
	  elecs.erase( cmp );
	} 
	else{
	  elecs.erase( ref );
	}
      }
    }
  }
  return;
}


reco::GenParticleCandidate TopElectronProducer::findTruth(const reco::CandidateCollection & parts, const TopElectronType & elec) {
  reco::GenParticleCandidate theGenElectron(0, reco::Particle::LorentzVector(0,0,0,0), reco::Particle::Point(0,0,0), 0, 0, true);
  for(unsigned int i=0; i!= pairGenRecoElectronsVector_.size(); i++){
    std::pair<const reco::Candidate*, TopElectronType*> pairGenRecoElectrons;
    pairGenRecoElectrons = pairGenRecoElectronsVector_[i];
    if(   fabs(elec.pt() - (pairGenRecoElectrons.second)->pt()) < 0.00001   ) {
      //cout << "elec.pt()  " << elec.pt()  << "   pairGenRecoElectrons.second->pt " << (pairGenRecoElectrons.second)->pt() << endl;
      reco::GenParticleCandidate aGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(pairGenRecoElectrons.first)));
      theGenElectron = aGenElectron;
    }
  }
  return theGenElectron;
}


void TopElectronProducer::matchTruth(const reco::CandidateCollection & particles, std::vector<TopElectronType> & electrons) {
  pairGenRecoElectronsVector_.clear();
  for(reco::CandidateCollection::const_iterator itGenElectron = particles.begin(); itGenElectron != particles.end(); ++itGenElectron) {
    reco::GenParticleCandidate aGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenElectron)));
    if (abs(aGenElectron.pdgId())==11 && aGenElectron.status()==1){
      TopElectronType* bestRecoElectron = 0;
      bool recoElectronFound = false;
      float bestDR = 100000;
      //loop over reconstructed electrons
      for (size_t e = 0; e < electrons.size(); ++e) {
	double recoEtOnGenEt = electrons[e].et()/aGenElectron.et();
	// if the charge is the same and the energy comparable
	//FIXME set recoEtOnGenEt cut configurable 
	  float currDR = DeltaR<reco::Candidate>()(aGenElectron, electrons[e]);
	  //if ( aGenElectron.charge()==electrons[e].charge() && recoEtOnGenEt > minRecoOnGenEt_ 
	  //     &&  recoEtOnGenEt < maxRecoOnGenEt_ && currDR < maxDeltaR_ ) {
	  if (  recoEtOnGenEt > minRecoOnGenEt_ 
		&&  recoEtOnGenEt < maxRecoOnGenEt_ && currDR < maxDeltaR_ ) {
	    //if the reco electrons is the closest one
	    if ( currDR < bestDR) {
	      bestRecoElectron = &electrons[e];
	      bestDR = currDR;
	      recoElectronFound = true;
	    }
	  }
      }
      if(recoElectronFound == true){
	pairGenRecoElectronsVector_.push_back(
					       std::pair<const reco::Candidate*,TopElectronType*>(&*itGenElectron, bestRecoElectron)
					       );
      }
    }
  }
}


double TopElectronProducer::electronID(const edm::Handle<std::vector<TopElectronType> > & elecs,
                                       const edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs, int idx) {
  //find elecID for elec with index idx
  edm::Ref<std::vector<TopElectronType> > elecsRef( elecs, idx );
  reco::ElectronIDAssociationCollection::const_iterator elecID = elecIDs->find( elecsRef );

  //return corresponding elecID (only 
  //cut based available at the moment)
  const reco::ElectronIDRef& id = elecID->val;
  return id->cutBasedDecision();
}
