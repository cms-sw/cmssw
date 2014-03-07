#ifndef PhysicsTools_PFCandProducer_GenJetClosestMatchSelectorDefinition
#define PhysicsTools_PFCandProducer_GenJetClosestMatchSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>

struct GenJetClosestMatchSelectorDefinition {


  typedef reco::GenJetCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::GenJet *> container;
  typedef container::const_iterator const_iterator;

  GenJetClosestMatchSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) {

    matchTo_ = iC.consumes< edm::View<reco::Candidate> >(cfg.getParameter< edm::InputTag >( "MatchTo" ));
  }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc,
	       const edm::Event & e,
	       const edm::EventSetup& s)
  {

    selected_.clear();

    edm::Handle< edm::View<reco::Candidate> > matchCandidates;
    e.getByToken( matchTo_, matchCandidates);


    unsigned key=0;

    //    std::cout<<"number of candidates "<<matchCandidates->size()<<std::endl;

    typedef edm::View<reco::Candidate>::const_iterator IC;
    for( IC ic = matchCandidates->begin();
	 ic!= matchCandidates->end(); ++ic ) {

      double eta2 = ic->eta();
      double phi2 = ic->phi();

      //      std::cout<<"cand "<<eta2<<" "<<phi2<<std::endl;


      // look for the closest gen jet
      double deltaR2Min = 9999;
      collection::const_iterator closest = hc->end();
      for( collection::const_iterator genjet = hc->begin();
	   genjet != hc->end();
	   ++genjet, ++key) {

	reco::GenJetRef genJetRef(hc, key);

	// is it matched?

	double eta1 = genjet->eta();
	double phi1 = genjet->phi();


	double deltaR2 = reco::deltaR2(eta1, phi1, eta2, phi2);

	// std::cout<<"  genjet "<<eta1<<" "<<phi1<<" "<<deltaR2<<std::endl;

	// cut should be a parameter
	if( deltaR2<deltaR2Min ) {
	  deltaR2Min = deltaR2;
	  closest = genjet;
	}
      }

      if(deltaR2Min<0.01 ) {
	// std::cout<<deltaR2Min<<std::endl;
	selected_.push_back( new reco::GenJet(*closest) );
      }
    } // end collection iteration

    // std::cout<<selected_.size()<<std::endl;
  } // end select()

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  edm::EDGetTokenT<edm::View<reco::Candidate> >  matchTo_;
};

#endif
