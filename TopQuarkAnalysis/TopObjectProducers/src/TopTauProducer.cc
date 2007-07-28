//
// Author:  Christophe Delaere
// Created: Thu Jul  26 11:08:00 CEST 2007
//
// $Id: $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <memory>

// specialization of the PtInverseComparator: TopTau is not (yet) a candidate.
template<>
bool PtInverseComparator<TopTau>::operator()( const TopTau & t1, const TopTau & t2 ) const {
  return t1.getJetTag()->jet().pt() > t2.getJetTag()->jet().pt();
}

//
// constructors and destructor
//

TopTauProducer::TopTauProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  tauSrc_           = iConfig.getParameter<edm::InputTag>("tauSource");
  doGenMatch_       = iConfig.getParameter<bool>         ("doGenMatch");
  addResolutions_   = iConfig.getParameter<bool>         ("addResolutions");
  addLRValues_      = iConfig.getParameter<bool>         ("addLRValues");
  genPartSrc_       = iConfig.getParameter<edm::InputTag>("genParticleSource");
  tauResoFile_      = iConfig.getParameter<std::string>  ("tauResoFile");
  tauLRFile_        = iConfig.getParameter<std::string>  ("tauLRFile");
  redoDiscriminant_ = iConfig.getParameter<bool>         ("redoDiscriminant");
  if(redoDiscriminant_) {
    Rmatch_          = iConfig.getParameter<double>       ("Rmatch");
    Rsig_            = iConfig.getParameter<double>       ("Rsig");
    Riso_            = iConfig.getParameter<double>       ("Riso");
    pT_LT_           = iConfig.getParameter<double>       ("pT_LT");
    pT_min_          = iConfig.getParameter<double>       ("pT_min");
  }

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new TopObjectResolutionCalc(tauResoFile_);
  }

  // produces vector of taus
  produces<std::vector<TopTau > >();
}


TopTauProducer::~TopTauProducer() {
  if (addResolutions_) delete theResoCalc_;
}


//
// member functions
//

void TopTauProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
 
  // Get the collection of taus from the event
  edm::Handle<std::vector<TopTauType> > taus;
  iEvent.getByLabel(tauSrc_, taus);

  // Get the vector of generated particles from the event if needed
  edm::Handle<reco::CandidateCollection> particles;
  if (doGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
  }

  // prepare LR calculation if required
  if (addLRValues_) {
    theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, "", tauLRFile_);
  }

  // loop over taus
  std::vector<TopTau> * topTaus = new std::vector<TopTau>(); 
  for (std::vector<TopTauType>::const_iterator tau = taus->begin();
       tau != taus->end(); ++tau) {
    // check the discriminant
    bool disc = redoDiscriminant_ ? tau->discriminator(Rmatch_,Rsig_,Riso_,pT_LT_,pT_min_) : tau->discriminator();
    if(!disc) continue;
    // construct the TopTau
    TopTau aTau(*tau);
    // match to generated final state taus
    if (doGenMatch_) {
      // initialize best match as null
      reco::GenParticleCandidate bestGenTau(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), 0, 0);
      float bestDR = 0;
      // find the closest generated tau. No charge cut is applied
      for (reco::CandidateCollection::const_iterator itGenTau = particles->begin(); itGenTau != particles->end(); ++itGenTau) {
        reco::GenParticleCandidate aGenTau = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenTau)));
        if (abs(aGenTau.pdgId())==15 && aGenTau.status()==2) {
	  float currDR = DeltaR<reco::Candidate>()(aGenTau, tau->getJetTag()->jet());
          if (bestDR == 0 || currDR < bestDR) {
            bestGenTau = aGenTau;
            bestDR = currDR;
          }
        }
      }
      aTau.setGenLepton(bestGenTau);
    }
    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(aTau);
    }
    // add top lepton id LR info if requested
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(aTau, iEvent);
    }
    // add the tau to the vector of TopTaus
    topTaus->push_back(TopTau(aTau));
  }

  // sort taus in pT
  std::sort(topTaus->begin(), topTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopTau> > pOutTau(topTaus);
  iEvent.put(pOutTau);

  // destroy the lepton LR calculator
  if (addLRValues_) delete theLeptonLRCalc_;

}

