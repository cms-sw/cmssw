//
// Author:  Christophe Delaere
// Created: Thu Jul  26 11:08:00 CEST 2007
//
// $Id: TopTauProducer.cc,v 1.11 2007/10/30 10:02:34 delaer Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <DataFormats/TauReco/interface/PFTau.h>
#include <DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h>
#include <DataFormats/TauReco/interface/CaloTau.h>
#include <DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h>

#include <vector>
#include <memory>

//
// constructors and destructor
//

TopTauProducer::TopTauProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  tauSrc_         = iConfig.getParameter<edm::InputTag>( "tauSource" );
  tauDiscSrc_     = iConfig.getParameter<edm::InputTag>( "tauDiscriminatorSource");
  addGenMatch_    = iConfig.getParameter<bool>         ( "addGenMatch" );
  addResolutions_ = iConfig.getParameter<bool>         ( "addResolutions" );
  useNNReso_      = iConfig.getParameter<bool>         ( "useNNResolutions" );
  addLRValues_    = iConfig.getParameter<bool>         ( "addLRValues" );
  genPartSrc_     = iConfig.getParameter<edm::InputTag>( "genParticleSource" );
  tauResoFile_    = iConfig.getParameter<std::string>  ( "tauResoFile" );
  tauLRFile_      = iConfig.getParameter<std::string>  ( "tauLRFile" );

  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new TopObjectResolutionCalc(edm::FileInPath(tauResoFile_).fullPath(), useNNReso_);
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
  edm::Handle<PFTauCollection> PFtaus;
  edm::Handle<PFTauDiscriminatorByIsolation> PFtauIsolator;
  edm::Handle<CaloTauCollection> Calotaus; 
  edm::Handle<CaloTauDiscriminatorByIsolation> CalotauIsolator;
  bool hasPFtaus = false;
  bool hasCalotaus = false;
  try {
    iEvent.getByLabel(tauSrc_, PFtaus);
    iEvent.getByLabel(tauDiscSrc_, PFtauIsolator);
    hasPFtaus = true;
  } catch( const edm::Exception &roEX) { }
  try {
    iEvent.getByLabel(tauSrc_, Calotaus);
    iEvent.getByLabel(tauDiscSrc_, CalotauIsolator);
    hasCalotaus = true;
  } catch( const edm::Exception &roEX) { }
  if(!hasCalotaus && !hasPFtaus) {
    //Important note:
    // We are not issuing a LogError to be able to run on AOD samples
    // produced < 1_7_0, like CSA07 samples.
    // Note that missing input will not block je job.
    // In that case, an empty collection will be produced.
    edm::LogWarning("DataSource") << "No Tau collection found.";
  }
  if(hasCalotaus && hasPFtaus) {
    edm::LogError("DataSource") << "Ambiguous datasource. Taus can be both CaloTaus or PF taus.";
  }

  // Get the vector of generated particles from the event if needed
  edm::Handle<reco::CandidateCollection> particles;
  if (addGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
  }

  // prepare LR calculation if required
  if (addLRValues_) {
    theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, "", "", edm::FileInPath(tauLRFile_).fullPath());
  }

  // collection of produced objects
  std::vector<TopTau> * topTaus = new std::vector<TopTau>(); 

  // loop over taus and prepare TopTaus
  if(hasPFtaus) {
    for (PFTauCollection::size_type iPFTau=0;iPFTau<PFtaus->size();iPFTau++) {
      // check the discriminant
      PFTauRef thePFTau(PFtaus,iPFTau);
      bool disc = (*PFtauIsolator)[thePFTau];
      if(!disc) continue;
      // construct the TopTau
      TopTau aTau(*thePFTau);
      // set the additional variables
      const reco::PFJet *pfJet = dynamic_cast<const reco::PFJet*>(thePFTau->pfTauTagInfoRef()->pfjetRef().get());
      if(pfJet) {
        aTau.setEmEnergyFraction(pfJet->chargedEmEnergyFraction()+pfJet->neutralEmEnergyFraction());
        aTau.setEoverP(thePFTau->energy()/thePFTau->leadTrack()->p());
      }
      // add the tau to the vector of TopTaus
      topTaus->push_back(TopTau(aTau));
    }
  } else if(hasCalotaus) {
    for (CaloTauCollection::size_type iCaloTau=0;iCaloTau<Calotaus->size();iCaloTau++) {
      // check the discriminant
      CaloTauRef theCaloTau(Calotaus,iCaloTau);
      bool disc = (*CalotauIsolator)[theCaloTau];
      if(!disc) continue;
      // construct the TopTau
      TopTau aTau(*theCaloTau);
      // set the additional variables
      const reco::CaloJet *tauJet = dynamic_cast<const reco::CaloJet*>(theCaloTau->caloTauTagInfoRef()->calojetRef().get());
      if(tauJet) {
        aTau.setEmEnergyFraction(tauJet->emEnergyFraction());
        aTau.setEoverP(tauJet->energy()/theCaloTau->leadTrack()->p());
      }
      // add the tau to the vector of TopTaus
      topTaus->push_back(TopTau(aTau));
    }
  }

  // loop on the resulting collection of TopTaus, and set other informations
  for(std::vector<TopTau>::iterator aTau = topTaus->begin();aTau<topTaus->end(); ++aTau) {
    // match to generated final state taus
    if (addGenMatch_) {
      // initialize best match as null
      reco::GenParticleCandidate bestGenTau(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), 0, 0, true);
      float bestDR = 5.; // this is the upper limit on the candidate matching. 
      // find the closest generated tau. No charge cut is applied
      for (reco::CandidateCollection::const_iterator itGenTau = particles->begin(); itGenTau != particles->end(); ++itGenTau) {
        reco::GenParticleCandidate aGenTau = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenTau)));
        if (abs(aGenTau.pdgId())==15 && aGenTau.status()==2) {
	  float currDR = DeltaR<reco::Candidate>()(aGenTau, *aTau);
          if (currDR < bestDR) {
            bestGenTau = aGenTau;
            bestDR = currDR;
          }
        }
      }
      aTau->setGenLepton(bestGenTau);
    }
    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(*aTau);
    }
    // add top lepton id LR info if requested
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(*aTau, iEvent);
    }
  }

  // sort taus in pT
  std::sort(topTaus->begin(), topTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopTau> > pOutTau(topTaus);
  iEvent.put(pOutTau);

  // destroy the lepton LR calculator
  if (addLRValues_) delete theLeptonLRCalc_;

}

