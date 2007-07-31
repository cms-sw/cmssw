//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.h,v 1.8 2007/07/12 19:36:00 lowette Exp $
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
  
  explicit TopElectronProducer(const edm::ParameterSet&);
  ~TopElectronProducer();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  
  void removeGhosts(std::vector<TopElectron>*);
  reco::GenParticleCandidate matchTruth(const reco::CandidateCollection&, const TopElectronType&);
  double electronID(edm::Handle<TopElectronTypeCollection>&, 
		    edm::Handle<reco::ElectronIDAssociationCollection>&, int);
  
 private:

  edm::InputTag src_, gen_, elecID_;
  bool useElecID_, useTrkIso_, useCalIso_, useResolution_;
  bool useLikelihood_, useGenMatching_, useGhostRemoval_;
  std::string resolutionInput_, likelihoodInput_;

  TopObjectResolutionCalc *resolution_;
  TopLeptonTrackerIsolationPt  *trkIsolation_;
  TopLeptonCaloIsolationEnergy *calIsolation_;
  TopLeptonLRCalc *likelihood_;
  
  PtInverseComparator<TopElectron> ptComparator_;
};

#endif
