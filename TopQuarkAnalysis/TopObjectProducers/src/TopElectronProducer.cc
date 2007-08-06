//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.cc,v 1.12 2007/07/31 21:59:17 rwolf Exp $
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
  src_   ( cfg.getParameter<edm::InputTag>( "src"    ) ),
  gen_   ( cfg.getParameter<edm::InputTag>( "gen"    ) ),
  elecID_( cfg.getParameter<edm::InputTag>( "elecID" ) ),
  useElecID_       ( cfg.getParameter<bool>( "useElectronID"  ) ),
  useTrkIso_       ( cfg.getParameter<bool>( "useTrkIsolation") ),
  useCalIso_       ( cfg.getParameter<bool>( "useCalIsolation") ),
  useResolution_   ( cfg.getParameter<bool>( "useResolutions" ) ),
  useLikelihood_   ( cfg.getParameter<bool>( "useLikelihood"  ) ),
  useGenMatching_  ( cfg.getParameter<bool>( "useGenMatching" ) ),
  useGhostRemoval_ ( cfg.getParameter<bool>( "useGhostRemoval") ),
  resolutionInput_( cfg.getParameter<std::string>( "resolutionInput" ) ),
  likelihoodInput_( cfg.getParameter<std::string>( "likelihoodInput" ) )
{
  if( useResolution_){
    resolution_= new TopObjectResolutionCalc( resolutionInput_);
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
      sel.setGenLepton( matchTruth( *parts, *elec ) );
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

reco::GenParticleCandidate
TopElectronProducer::matchTruth(const reco::CandidateCollection& parts, const TopElectronType& elec)
{
  float minR = 0.;
  reco::GenParticleCandidate gene;
  reco::CandidateCollection::const_iterator part = parts.begin();
  for( ; part!=parts.end(); ++part ){
    if( (abs(part->pdgId())==11) && (part->status()==1) && part->charge()==elec.charge() ){
      float dr = DeltaR<reco::Candidate>()( *part, elec );
      if( minR==0 || dr<minR ){
	minR = dr;
	gene = dynamic_cast<const reco::GenParticleCandidate&>(*part);
      }
    }
  }
  return gene;
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

