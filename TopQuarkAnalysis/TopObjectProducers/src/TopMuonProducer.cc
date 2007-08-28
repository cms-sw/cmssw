//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopMuonProducer.cc,v 1.10 2007/08/27 11:04:34 tsirig Exp $
//

#include <vector>
#include <memory>

#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonTrackerIsolationPt.h"
#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonCaloIsolationEnergy.h"
#include "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"

TopMuonProducer::TopMuonProducer(const edm::ParameterSet & cfg):
  muonSrc_   ( cfg.getParameter<edm::InputTag>( "muonSource" ) ),
  genPartSrc_( cfg.getParameter<edm::InputTag>( "genParticleSource" ) ),
  tracksTag_ ( cfg.getParameter<edm::InputTag>( "tracks" ) ),
  useTrkIso_ ( cfg.getParameter<bool>( "useTrkIsolation") ),
  useCalIso_ ( cfg.getParameter<bool>( "useCalIsolation") ),
  addResolutions_( cfg.getParameter<bool>( "addResolutions" ) ),
  useNNReso_     ( cfg.getParameter<bool>( "useNNresolution") ),
  addLRValues_   ( cfg.getParameter<bool>( "addLRValues" ) ),
  doGenMatch_    ( cfg.getParameter<bool>( "doGenMatch" ) ),
  muonResoFile_  ( cfg.getParameter<std::string>( "muonResoFile" ) ),
  muonLRFile_    ( cfg.getParameter<std::string>( "muonLRFile" ) ),
  minRecoOnGenEt_( cfg.getParameter<double>( "minRecoOnGenEt" ) ),
  maxRecoOnGenEt_( cfg.getParameter<double>( "maxRecoOnGenEt" ) ),
  maxDeltaR_     ( cfg.getParameter<double>( "maxDeltaR" ) )
{
  if( addResolutions_){
    theResoCalc_ = new TopObjectResolutionCalc( muonResoFile_, useNNReso_ );
  }
  produces<std::vector<TopMuon > >();
}

TopMuonProducer::~TopMuonProducer() {
  if ( addResolutions_) 
    delete theResoCalc_;
}

void 
TopMuonProducer::produce(edm::Event & evt, const edm::EventSetup & setup)
{ 
  edm::Handle<std::vector<TopMuonType> > muons;
  evt.getByLabel(muonSrc_, muons);
  
  edm::Handle<reco::CandidateCollection> parts;
  if (doGenMatch_) {
    evt.getByLabel(genPartSrc_, parts);
  }

  TopMuonTypeCollection myMuons = *muons;
  if ( doGenMatch_) {
    matchTruth(*parts, myMuons ) ;
  }

  //prepare isolation calculation
  if( useTrkIso_) trkIsolation_= new TopLeptonTrackerIsolationPt ( setup );
  if( useCalIso_) calIsolation_= new TopLeptonCaloIsolationEnergy( setup );

  // prepare LR calculation
  if (addLRValues_) {
    theLeptonLRCalc_ = new TopLeptonLRCalc( setup, "", muonLRFile_, "", tracksTag_);
  }

  std::vector<TopMuon>* selected = new std::vector<TopMuon>(); 
  TopMuonTypeCollection::const_iterator muon = muons->begin();
  for( ;muon!=muons->end(); ++muon){
    TopMuon sel( *muon );

    if( useTrkIso_){
      sel.setTrackIso( trkIsolation_->calculate( sel, evt ) );
    }
    
    if( useCalIso_){
      sel.setCaloIso ( calIsolation_->calculate( sel, evt ) );
    }

    if( addResolutions_){
      (*theResoCalc_)( sel );
    }

    if( addLRValues_){
      theLeptonLRCalc_->calcLikelihood( sel, evt );
    }
    
    if ( doGenMatch_) {
      sel.setGenLepton( findTruth( *parts, *muon ) );
    }

    //add sel to selected
    selected->push_back( TopMuon( sel ) );
  }

  // sort muons in pt
  std::sort( selected->begin(), selected->end(), pTComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopMuon> > ptr( selected );
  evt.put( ptr );

  if( useTrkIso_ ) 
    delete trkIsolation_;
  if( useCalIso_ ) 
    delete calIsolation_;  
  if( addLRValues_) 
    delete theLeptonLRCalc_;
}

reco::GenParticleCandidate
TopMuonProducer::findTruth(const reco::CandidateCollection& parts, const TopMuonType& muon)
{
  reco::GenParticleCandidate gen;
  for(unsigned int idx=0; idx!=pairGenRecoMuonsVector_.size(); ++idx){
    std::pair<const reco::Candidate*, TopMuonType*> pairGenRecoMuons;
    pairGenRecoMuons = pairGenRecoMuonsVector_[idx];
    double dR = DeltaR<reco::Candidate>()( muon, *(pairGenRecoMuons.second));
    if( !(dR > 0) ){
      gen = *(dynamic_cast<const reco::GenParticleCandidate*>( pairGenRecoMuons.first ) );
    }
  }
  return gen;
}

void 
TopMuonProducer::matchTruth( const reco::CandidateCollection& parts, TopMuonTypeCollection& muons )
{
  pairGenRecoMuonsVector_.clear();
  reco::CandidateCollection::const_iterator part = parts.begin();
  for( ; part != parts.end(); ++part ){
    reco::GenParticleCandidate gen = *(dynamic_cast<const reco::GenParticleCandidate*>( &(*part)) );
    if( abs(gen.pdgId())==13 && gen.status()==1 ){
      bool   found = false;
      double minDR = 99999;
      TopMuonType* rec = 0;
      TopMuonTypeCollection::iterator muon = muons.begin();
      for ( ; muon !=muons.end(); ++muon){
	double dR = DeltaR<reco::Candidate>()( gen, *muon);
	double ptRecOverGen = muon->pt()/gen.pt();
	if ( ( ptRecOverGen > minRecoOnGenEt_ ) && 
	     ( ptRecOverGen < maxRecoOnGenEt_ ) && 
	     ( dR < maxDeltaR_) ){
	  if ( dR < minDR ){
	    rec = &(*muon);
	    minDR = dR;
	    found = true;
	  }
	}
      }
      if( found == true ){
	pairGenRecoMuonsVector_.push_back( std::pair<const reco::Candidate*,TopMuonType*>(&(*part), rec ) );
      }
    }
  }
}
