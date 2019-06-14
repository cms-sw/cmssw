// -*- C++ -*-
//
// Package:    V0Producer
// Class:      V0Producer
//
/**\class V0Producer V0Producer.cc MyProducers/V0Producer/src/V0Producer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Fri May 18 22:57:40 CEST 2007
//
//

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
//#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

#include "V0Fitter.h"

class dso_hidden V0Producer final : public edm::stream::EDProducer<> {
public:
  explicit V0Producer(const edm::ParameterSet&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  V0Fitter theVees;
};

// Constructor
V0Producer::V0Producer(const edm::ParameterSet& iConfig) : theVees(iConfig, consumesCollector()) {
  produces<reco::VertexCompositeCandidateCollection>("Kshort");
  produces<reco::VertexCompositeCandidateCollection>("Lambda");
  //produces< reco::VertexCompositeCandidateCollection >("LambdaBar");
}

//
// Methods
//

// Producer Method
void V0Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Create auto_ptr for each collection to be stored in the Event
  auto kShortCandidates = std::make_unique<reco::VertexCompositeCandidateCollection>();

  auto lambdaCandidates = std::make_unique<reco::VertexCompositeCandidateCollection>();

  // invoke the fitter which reconstructs the vertices and fills,
  //  collections of Kshorts, Lambda0s
  theVees.fitAll(iEvent, iSetup, *kShortCandidates, *lambdaCandidates);

  // Write the collections to the Event
  kShortCandidates->shrink_to_fit();
  iEvent.put(std::move(kShortCandidates), std::string("Kshort"));
  lambdaCandidates->shrink_to_fit();
  iEvent.put(std::move(lambdaCandidates), std::string("Lambda"));
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(V0Producer);
//DEFINE_FWK_MODULE(V0finder);
