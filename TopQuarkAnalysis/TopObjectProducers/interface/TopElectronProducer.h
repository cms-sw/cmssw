//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.h,v 1.11 2007/08/20 15:36:08 lowette Exp $
//

#ifndef TopObjectProducers_TopElectronProducer_h
#define TopObjectProducers_TopElectronProducer_h

#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"

#include "AnalysisDataFormats/Egamma/interface/ElectronID.h"
#include "AnalysisDataFormats/Egamma/interface/ElectronIDAssociation.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"


class TopObjectResolutionCalc;
class TopLeptonTrackerIsolationPt;
class TopLeptonCaloIsolationEnergy;
class TopLeptonLRCalc;


class TopElectronProducer : public edm::EDProducer {
  
 public:
  
  explicit TopElectronProducer(const edm::ParameterSet & cfg);
  ~TopElectronProducer();  
  virtual void produce(edm::Event&, const edm::EventSetup & setup);
  
 private:
  
  reco::GenParticleCandidate findTruth(const reco::CandidateCollection & parts, const TopElectronType & elec);
  void matchTruth(const reco::CandidateCollection & particles, TopElectronTypeCollection & electrons);
  double electronID(edm::Handle<TopElectronTypeCollection> & elecs, 
		    edm::Handle<reco::ElectronIDAssociationCollection> & elecIDs, int idx);
  void removeGhosts(std::vector<TopElectron> * elecs);
  
 private:

  edm::InputTag src_, gen_, elecID_, tracksTag_;
  bool useElecID_, useTrkIso_, useCalIso_, useResolution_;
  bool useLikelihood_, useGenMatching_, useGhostRemoval_;
  std::string resolutionInput_, likelihoodInput_;
  double minRecoOnGenEt_, maxRecoOnGenEt_, maxDeltaR_;

  TopObjectResolutionCalc *resolution_;
  TopLeptonTrackerIsolationPt  *trkIsolation_;
  TopLeptonCaloIsolationEnergy *calIsolation_;
  TopLeptonLRCalc *likelihood_;
  std::vector<std::pair<const reco::Candidate *, TopElectronType*> > pairGenRecoElectronsVector_;
  GreaterByPt<TopElectron> ptComparator_;

};

#endif
