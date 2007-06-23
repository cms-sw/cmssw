//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.cc,v 1.7 2007/06/14 19:34:24 jlamb Exp $
//

#include "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"

#include "TopQuarkAnalysis/TopLeptonSelection/interface/TopLeptonLRCalc.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <memory>


//
// constructors and destructor
//

TopElectronProducer::TopElectronProducer(const edm::ParameterSet & iConfig) {
  // initialize the configurables
  electronSrc_      = iConfig.getParameter<edm::InputTag>("electronSource");
  doGenMatch_       = iConfig.getParameter<bool>         ("doGenMatch");
  addResolutions_   = iConfig.getParameter<bool>         ("addResolutions");
  addLRValues_      = iConfig.getParameter<bool>         ("addLRValues");
  genPartSrc_       = iConfig.getParameter<edm::InputTag>("genParticleSource");
  electronResoFile_ = iConfig.getParameter<std::string>  ("electronResoFile");
  electronLRFile_   = iConfig.getParameter<std::string>  ("electronLRFile");
  
  // construct resolution calculator
  if (addResolutions_) {
    theResoCalc_ = new TopObjectResolutionCalc(electronResoFile_);
  }

  // produces vector of electrons
  produces<std::vector<TopElectron > >();
}


TopElectronProducer::~TopElectronProducer() {
  if (addResolutions_) delete theResoCalc_;
}


//
// member functions
//

void TopElectronProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
 
  // Get the collection of electrons from the event
  edm::Handle<std::vector<TopElectronType> > electronsHandle; 
  iEvent.getByLabel(electronSrc_, electronsHandle);
  std::vector<TopElectronType> electrons=*electronsHandle;

  //remove any duplicate electrons that might be in the event
  electrons=removeEleDupes(electrons);

  // Get the vector of generated particles from the event if needed
  edm::Handle<reco::CandidateCollection> particles;
  if (doGenMatch_) {
    iEvent.getByLabel(genPartSrc_, particles);
  }

  // prepare LR calculation if required
  if (addLRValues_) {
    theLeptonLRCalc_ = new TopLeptonLRCalc(iSetup, electronLRFile_, "");
  }

  // loop over electrons
  std::vector<TopElectron> * topElectrons = new std::vector<TopElectron>(); 
  for (size_t e = 0; e < electrons.size(); ++e) {
    // construct the TopElectron
    TopElectron anElectron(electrons[e]);
    // match to generated final state electrons
    if (doGenMatch_) {
      // initialize best match as null
      reco::GenParticleCandidate bestGenElectron(0, reco::Particle::LorentzVector(0,0,0,0), reco::Particle::Point(0,0,0), 0, 0);
      float bestDR = 0;
      // find the closest generated electron
      for(reco::CandidateCollection::const_iterator itGenElectron = particles->begin(); itGenElectron != particles->end(); ++itGenElectron) {
        reco::GenParticleCandidate aGenElectron = *(dynamic_cast<reco::GenParticleCandidate *>(const_cast<reco::Candidate *>(&*itGenElectron)));
        if (abs(aGenElectron.pdgId())==11 && aGenElectron.status()==1 && aGenElectron.charge()==electrons[e].charge()) {
          float currDR = DeltaR<reco::Candidate>()(aGenElectron, electrons[e]);
          if (bestDR == 0 || currDR < bestDR) {
            bestGenElectron = aGenElectron;
            bestDR = currDR;
          }
        }
        anElectron.setGenLepton(bestGenElectron);
      }
    }
    // add resolution info if demanded
    if (addResolutions_) {
      (*theResoCalc_)(anElectron);
    }
    // add top lepton id LR info if requested
    if (addLRValues_) {
      theLeptonLRCalc_->calcLikelihood(anElectron, iEvent);
    }
    // add the electron to the vector of TopElectrons
    topElectrons->push_back(TopElectron(anElectron));
  }

  // sort electrons in pT
  std::sort(topElectrons->begin(), topElectrons->end(), pTElectronComparator_);

  // put genEvt object in Event
  std::auto_ptr<std::vector<TopElectron> > pOutElectron(topElectrons);
  iEvent.put(pOutElectron);

  // destroy the lepton LR calculator
  if (addLRValues_) delete theLeptonLRCalc_;

}

//it is possible that there are multiple electron objects in the collection that correspond to the same
//real physics object - a supercluster with two tracks reconstructed to it, or a track that points to two different SC
// (i would guess the latter doesn't actually happen).  Mostly they should be removed by the regular selection but...
//this function removes these duplicate electrons from the selected electrons vector (and returns the new vector, the input
//is unmodified).  
std::vector<TopElectronType> TopElectronProducer::removeEleDupes(const std::vector<TopElectronType> &electrons) {
  
  //contains indices of duplicate electrons marked for removal
  //I do it this way because removal during the loop is more confusing
  std::vector<size_t> indicesToRemove;
  
  for (size_t ie=0;ie<electrons.size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) continue;//ignore if already marked for removal
    
    reco::GsfTrackRef thistrack=electrons[ie].gsfTrack();
    reco::SuperClusterRef thissc=electrons[ie].superCluster();

    for (size_t je=ie+1;je<electrons.size();je++) {
      if (find(indicesToRemove.begin(),indicesToRemove.end(),je)!=indicesToRemove.end()) continue;//ignore if already marked for removal

      if ((thistrack==electrons[je].gsfTrack()) ||
	  (thissc==electrons[je].superCluster()) ) {//we have a match, arbitrate and mark one for removal
	//keep the one with E/P closer to unity
	float diff1=fabs(electrons[ie].eSuperClusterOverP()-1);
	float diff2=fabs(electrons[je].eSuperClusterOverP()-1);
	
	if (diff1<diff2) {
	  indicesToRemove.push_back(je);
	} else {
	  indicesToRemove.push_back(ie);
	}
      }
    }
  }
  std::vector<TopElectronType> output;
  //now remove the ones marked
  for (size_t ie=0;ie<electrons.size();ie++) {
    if (find(indicesToRemove.begin(),indicesToRemove.end(),ie)!=indicesToRemove.end()) {
      continue;
    } else {
      output.push_back(electrons[ie]);
    }
  }
  
  
  return output;
}

