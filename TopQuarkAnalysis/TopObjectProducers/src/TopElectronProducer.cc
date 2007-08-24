//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.cc,v 1.15 2007/08/22 13:29:23 rwolf Exp $
//

#include <vector>
#include <memory>

#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonTrackerIsolationPt.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonCaloIsolationEnergy.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"


TopElectronProducer::TopElectronProducer(const edm::ParameterSet & cfg):
  src_   ( cfg.getParameter<edm::InputTag>( "electronSource" ) ),
  gen_   ( cfg.getParameter<edm::InputTag>( "genParticleSource" ) ),
  elecID_( cfg.getParameter<edm::InputTag>( "electronIDSource" ) ),
  useElecID_       ( cfg.getParameter<bool>( "useElectronID"  ) ),
  useTrkIso_       ( cfg.getParameter<bool>( "useTrkIsolation") ),
  useCalIso_       ( cfg.getParameter<bool>( "useCalIsolation") ),
  useResolution_   ( cfg.getParameter<bool>( "addResolutions" ) ),
  useLikelihood_   ( cfg.getParameter<bool>( "addLRValues"  ) ),
  useGenMatching_  ( cfg.getParameter<bool>( "doGenMatch" ) ),
  useGhostRemoval_ ( cfg.getParameter<bool>( "removeDuplicates") ),
  resolutionInput_( cfg.getParameter<std::string>( "electronResoFile" ) ),
  likelihoodInput_( cfg.getParameter<std::string>( "electronLRFile" ) ),
  minRecoOnGenEt_ (cfg.getParameter<double>       ("minRecoOnGenEt") ),
  maxRecoOnGenEt_ (cfg.getParameter<double>       ("maxRecoOnGenEt") ),
  maxDeltaR_      (cfg.getParameter<double>       ("maxDeltaR") )
{
  if( useResolution_){
    resolution_= new TopObjectResolutionCalc( resolutionInput_,cfg.getParameter<bool>("useNNresolution"));
  }
  produces<std::vector<TopElectron> >();
}

TopElectronProducer::~TopElectronProducer() 
{
  if( useResolution_) 
    delete resolution_;
}

void 
TopElectronProducer::produce(edm::Event& evt, const edm::EventSetup& setup) 
{ 
  edm::Handle<TopElectronTypeCollection> elecs; 
  evt.getByLabel(src_, elecs);

  edm::Handle<reco::CandidateCollection> parts;
  if( useGenMatching_) evt.getByLabel(gen_, parts);

  edm::Handle<reco::ElectronIDAssociationCollection> elecIDs;
  if( useElecID_) evt.getByLabel( elecID_, elecIDs );

  //TopElectronTypeCollection electrons;
  TopElectronTypeCollection electrons = *elecs;
  if ( useGenMatching_) {
    matchTruth(*parts, electrons ) ;
  }
  
  
  //prepare isolation calculation
  if( useTrkIso_) trkIsolation_= new TopLeptonTrackerIsolationPt ( setup );
  if( useCalIso_) calIsolation_= new TopLeptonCaloIsolationEnergy( setup );
  
  //prepare LR calculation
  if( useLikelihood_) 
    likelihood_= new TopLeptonLRCalc( setup, likelihoodInput_, "" );

  std::vector<TopElectron>* selected = new std::vector<TopElectron>();
  TopElectronTypeCollection::const_iterator elec = elecs->begin();
  for(int idx=0; elec!=elecs->end(); ++elec, ++idx){
    TopElectron sel( *elec );

    if( useElecID_){
      sel.setLeptonID( electronID(elecs, elecIDs, idx) );
    }

    if( useTrkIso_){
      sel.setTrackIso( trkIsolation_->calculate( sel, evt ) );
    }
    
    if( useCalIso_){
      sel.setCaloIso ( calIsolation_->calculate( sel, evt ) );
    }

    if( useResolution_){
      (*resolution_)( sel );
    }

    if( useLikelihood_){
      likelihood_->calcLikelihood( sel, evt );
    }
    
    if ( useGenMatching_) {
      sel.setGenLepton( findTruth( *parts, *elec ) );
    }
    
    //add sel to selected
    selected->push_back( TopElectron( sel ) );
  }

  //sort electrons in pt
  std::sort( selected->begin(), selected->end(), ptComparator_);
  
  //remove ghosts if requested
  if( useGhostRemoval_){
    removeGhosts( selected );
  }
 
  //add selected to the event output and clean up
  std::auto_ptr<std::vector<TopElectron> > ptr( selected );
  evt.put( ptr );

  if( useTrkIso_ ) 
    delete trkIsolation_;
  if( useCalIso_ ) 
    delete calIsolation_;
  if( useLikelihood_) 
    delete likelihood_;
}



//get gen
reco::GenParticleCandidate
TopElectronProducer::findTruth(const reco::CandidateCollection& parts, const TopElectronType& elec)
{

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


//FindGen
void 
TopElectronProducer::matchTruth(const reco::CandidateCollection& particles, TopElectronTypeCollection& electrons)
{
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


double
TopElectronProducer::electronID(edm::Handle<TopElectronTypeCollection>& elecs,
				edm::Handle<reco::ElectronIDAssociationCollection>& elecIDs, int idx)
{
  //find elecID for elec with index idx
  edm::Ref<TopElectronTypeCollection> elecsRef( elecs, idx );
  reco::ElectronIDAssociationCollection::const_iterator elecID = elecIDs->find( elecsRef );

  //return corresponding elecID (only 
  //cut based available at the moment)
  const reco::ElectronIDRef& id = elecID->val;
  return id->cutBasedDecision();
}



void
TopElectronProducer::removeGhosts(std::vector<TopElectron>* elecs) 
{
  std::vector<TopElectron>::iterator cmp = elecs->begin();  
  std::vector<TopElectron>::iterator ref = elecs->begin();  
  for( ; ref<elecs->end(); ++ref ){
    for( ; (cmp!=ref) && cmp<elecs->end(); ++cmp ){
      if ((cmp->gsfTrack()==ref->gsfTrack()) || (cmp->superCluster()==ref->superCluster()) ){
	//same track or super cluster is used
	//keep the one with E/p closer to one	
	if(fabs(ref->eSuperClusterOverP()-1.) < fabs(cmp->eSuperClusterOverP()-1.)){
	  elecs->erase( cmp );
	} 
	else{
	  elecs->erase( ref );
	}
      }
    }
  }
  return;
}

