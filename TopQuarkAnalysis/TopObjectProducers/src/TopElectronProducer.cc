//
// $Id: TopElectronProducer.cc,v 1.29 2007/12/14 13:52:02 jlamb Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
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
  addGenMatch_      = iConfig.getParameter<bool>         ( "addGenMatch" );
  genPartSrc_       = iConfig.getParameter<edm::InputTag>( "genParticleSource" );
  maxDeltaR_        = iConfig.getParameter<double>       ( "maxDeltaR" );
  minRecoOnGenEt_   = iConfig.getParameter<double>       ( "minRecoOnGenEt" );
  maxRecoOnGenEt_   = iConfig.getParameter<double>       ( "maxRecoOnGenEt" );
  // resolution configurables
  addResolutions_   = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_        = iConfig.getParameter<bool>         ( "useNNResolutions" );
  electronResoFile_ = iConfig.getParameter<std::string>  ( "electronResoFile" );
  // isolation configurables
  addTrkIso_        = iConfig.getParameter<bool>         ( "addTrkIsolation" );
  tracksSrc_        = iConfig.getParameter<edm::InputTag>( "tracksSource" );
  addCalIso_        = iConfig.getParameter<bool>         ( "addCalIsolation" );
  towerSrc_         = iConfig.getParameter<edm::InputTag>( "towerSource" );
  // electron ID configurables
  addElecID_        = iConfig.getParameter<bool>         ( "addElectronID" );
  elecIDSrc_        = iConfig.getParameter<edm::InputTag>( "electronIDSource" );
  addElecIDRobust_  = iConfig.getParameter<bool>         ( "addElectronIDRobust" );
  elecIDRobustSrc_  = iConfig.getParameter<edm::InputTag>( "electronIDRobustSource" );
  
  // likelihood ratio configurables
  addLRValues_      = iConfig.getParameter<bool>         ( "addLRValues" );
  electronLRFile_   = iConfig.getParameter<std::string>  ( "electronLRFile" );
  // configurables for isolation from egamma producer
  addEgammaIso_     = iConfig.getParameter<bool>         ( "addEgammaIso");
  egammaTkIsoSrc_   = iConfig.getParameter<edm::InputTag>( "egammaTkIsoSource");
  egammaTkNumIsoSrc_= iConfig.getParameter<edm::InputTag>( "egammaTkNumIsoSource");
  egammaEcalIsoSrc_ = iConfig.getParameter<edm::InputTag>( "egammaEcalIsoSource");
  egammaHcalIsoSrc_ = iConfig.getParameter<edm::InputTag>( "egammaHcalIsoSource");
  
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

  // Get the collection of electrons from the event
  edm::Handle<std::vector<TopElectronType> > electronsHandle;
  iEvent.getByLabel(electronSrc_, electronsHandle);
  std::vector<TopElectronType> electrons = *electronsHandle;


  // prepare the MC matching
  edm::Handle<reco::GenParticleCollection> particles;
  if (addGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
    matchTruth(*particles, electrons) ;
  }

  // prepare isolation calculation
  edm::Handle<reco::TrackCollection> trackHandle;
  if (addTrkIso_) {
    trkIsolation_= new TrackerIsolationPt();
    iEvent.getByLabel(tracksSrc_, trackHandle);
  }
  edm::Handle<reco::PMGsfElectronIsoCollection> tkIsoHandle;
  edm::Handle<reco::PMGsfElectronIsoNumCollection> tkNumIsoHandle;
  edm::Handle<CandViewDoubleAssociations> ecalIsoHandle;
  edm::Handle<CandViewDoubleAssociations> hcalIsoHandle;
  if (addEgammaIso_) {
    iEvent.getByLabel(egammaTkIsoSrc_,tkIsoHandle);
    iEvent.getByLabel(egammaTkNumIsoSrc_,tkNumIsoHandle);
    iEvent.getByLabel(egammaEcalIsoSrc_,ecalIsoHandle);
    iEvent.getByLabel(egammaHcalIsoSrc_,hcalIsoHandle);
  }
  std::vector<CaloTower> towers;
  if (addCalIso_) {
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
  edm::Handle<reco::ElectronIDAssociationCollection> elecIDRobusts;
  if (addElecIDRobust_) iEvent.getByLabel(elecIDRobustSrc_, elecIDRobusts);
  
  // prepare LR calculation
  if(addLRValues_) {
    theLeptonLRCalc_= new TopLeptonLRCalc(iSetup, edm::FileInPath(electronLRFile_).fullPath(), "", "");
  }

  std::vector<TopElectron> * topElectrons = new std::vector<TopElectron>();
  for (size_t e = 0; e < electrons.size(); ++e) {
    // construct the TopElectron
    TopElectron anElectron(electrons[e]);
    // match to generated final state electrons
    if (addGenMatch_) {
      anElectron.setGenLepton(findTruth(*particles, electrons[e]));
    }
    // add resolution info
    if(addResolutions_){
      (*theResoCalc_)(anElectron);
    }
    // do tracker isolation
    if (addTrkIso_) {
      anElectron.setTrackIso(trkIsolation_->calculate(anElectron, *trackHandle));
    }
    // do calorimeter isolation
    if (addCalIso_) {
      anElectron.setCaloIso(calIsolation_->calculate(anElectron, towers));
    }
    // add isolation from egamma producers
    if (addEgammaIso_) {
      setEgammaIso(anElectron,electronsHandle,tkIsoHandle,tkNumIsoHandle,ecalIsoHandle,hcalIsoHandle,e);
    }
    // add electron ID info
    if (addElecID_) {
      anElectron.setLeptonID(electronID(electronsHandle, elecIDs, e));
    }
    if (addElecIDRobust_) {
      anElectron.setElectronIDRobust(electronID(electronsHandle, elecIDRobusts, e));
    }
    // add lepton LR info
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(anElectron, trackHandle, iEvent);
    }
    // add sel to selected
    topElectrons->push_back(TopElectron(anElectron));
  }

  
  // remove ghosts
  if (doGhostRemoval_) {
    removeEleDupes(topElectrons);
    //removeGhosts(electrons);has bug, replaced with clunkier but working code.
  }


  // sort electrons in pt
  std::sort(topElectrons->begin(), topElectrons->end(), pTComparator_);

  // add the electrons to the event output
  std::auto_ptr<std::vector<TopElectron> > ptr(topElectrons);
  iEvent.put(ptr);

  // clean up
  if (addTrkIso_) delete trkIsolation_;
  if (addCalIso_) delete calIsolation_;
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


reco::GenParticle TopElectronProducer::findTruth(const reco::GenParticleCollection & parts, const TopElectronType & elec) {
  reco::GenParticle theGenElectron(0, reco::Particle::LorentzVector(0,0,0,0), reco::Particle::Point(0,0,0), 0, 0, true);
  for(unsigned int i=0; i!= pairGenRecoElectronsVector_.size(); i++){
    std::pair<const reco::GenParticle*, TopElectronType*> pairGenRecoElectrons;
    pairGenRecoElectrons = pairGenRecoElectronsVector_[i];
    if(   fabs(elec.pt() - (pairGenRecoElectrons.second)->pt()) < 0.00001   ) {
      reco::GenParticle aGenElectron = *(dynamic_cast<reco::GenParticle *>(const_cast<reco::GenParticle *>(pairGenRecoElectrons.first)));
      theGenElectron = aGenElectron;
    }
  }
  return theGenElectron;
}


void TopElectronProducer::matchTruth(const reco::GenParticleCollection & particles, std::vector<TopElectronType> & electrons) {
  pairGenRecoElectronsVector_.clear();
  for(reco::GenParticleCollection::const_iterator itGenElectron = particles.begin(); itGenElectron != particles.end(); ++itGenElectron) {
    reco::GenParticle aGenElectron = *(dynamic_cast<reco::GenParticle *>(const_cast<reco::GenParticle *>(&*itGenElectron)));
    if (abs(aGenElectron.pdgId())==11 && aGenElectron.status()==1){
      TopElectronType* bestRecoElectron = 0;
      bool recoElectronFound = false;
      float bestDR = 100000;
      //loop over reconstructed electrons
      for (size_t e = 0; e < electrons.size(); ++e) {
	double recoEtOnGenEt = electrons[e].et()/aGenElectron.et();
	// if the charge is the same and the energy comparable
	//FIXME set recoEtOnGenEt cut configurable 
	  float currDR = DeltaR<reco::GenParticle, reco::Candidate>()(aGenElectron, electrons[e]);
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
					       std::pair<const reco::GenParticle*,TopElectronType*>(&*itGenElectron, bestRecoElectron)
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


//fill the TopElectron with the isolation quantities calculated by the egamma producers
void TopElectronProducer::setEgammaIso(TopElectron &anElectron,
				       const edm::Handle<std::vector<TopElectronType> > & elecs,
				       const edm::Handle<reco::PMGsfElectronIsoCollection> tkIsoHandle,
				       const edm::Handle<reco::PMGsfElectronIsoNumCollection> tkNumIsoHandle,
				       const edm::Handle<reco::CandViewDoubleAssociations> ecalIsoHandle,
				       const edm::Handle<reco::CandViewDoubleAssociations> hcalIsoHandle,
				       int idx) {


  //find isolations for elec with index idx
  edm::Ref<std::vector<TopElectronType> > elecsRef( elecs, idx );
  reco::CandidateBaseRef candRef(elecsRef);
  //reco::ElectronIDAssociationCollection::const_iterator elecID = elecIDs->find( elecsRef );
  anElectron.setEgammaTkIso((*tkIsoHandle)[elecsRef]);
  anElectron.setEgammaTkNumIso((*tkNumIsoHandle)[elecsRef]);
  anElectron.setEgammaEcalIso((*ecalIsoHandle)[candRef]);
  anElectron.setEgammaHcalIso((*hcalIsoHandle)[candRef]);
  
}

//it is possible that there are multiple electron objects in the collection that correspond to the same
//real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
// (i would guess the latter doesn't actually happen).
//NB triplicates also appear in the electron collection provided by egamma group, it is necessary to handle those correctly

//this function removes the duplicates/triplicates/multiplicates from the input vector
void TopElectronProducer::removeEleDupes(std::vector<TopElectron> *electrons) {
  
  //contains indices of duplicate electrons marked for removal
  //I do it this way because removal during the loop is more confusing
  std::vector<size_t> indicesToRemove;
  
  for (size_t ie=0;ie<electrons->size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) continue;//ignore if already marked for removal
    
    reco::GsfTrackRef thistrack=electrons->at(ie).gsfTrack();
    reco::SuperClusterRef thissc=electrons->at(ie).superCluster();

    for (size_t je=ie+1;je<electrons->size();je++) {
      if (find(indicesToRemove.begin(),indicesToRemove.end(),je)!=indicesToRemove.end()) continue;//ignore if already marked for removal
      
      if ((thistrack==electrons->at(je).gsfTrack()) ||
	  (thissc==electrons->at(je).superCluster()) ) {//we have a match, arbitrate and mark one for removal
	//keep the one with E/P closer to unity
	float diff1=fabs(electrons->at(ie).eSuperClusterOverP()-1);
	float diff2=fabs(electrons->at(je).eSuperClusterOverP()-1);
	
	if (diff1<diff2) {
	  indicesToRemove.push_back(je);
	} else {
	  indicesToRemove.push_back(ie);
	}
      }
    }
  }
  //now remove the ones marked
  //or in fact, copy the old vector into a tmp vector, skipping the ones that are duplicates,
  //then clear the original and copy back the contents of the tmp
  //this is ugly but it will work
  std::vector<TopElectron> tmp;
  for (size_t ie=0;ie<electrons->size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) {
      continue;
    } else {
      tmp.push_back(electrons->at(ie));
    }
  }
  //copy back
  electrons->clear();
  electrons->assign(tmp.begin(),tmp.end());
  
  return;
}



