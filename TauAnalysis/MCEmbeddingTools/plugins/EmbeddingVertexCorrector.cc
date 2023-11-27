// -*- C++ -*-
//
// Package:    TauAnalysis/EmbeddingProducer
// Class:      EmbeddingVertexCorrector
//
/**\class EmbeddingVertexCorrector EmbeddingVertexCorrector.cc TauAnalysis/EmbeddingProducer/plugins/EmbeddingVertexCorrector.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Artur Akhmetshin
//         Created:  Sat, 23 Apr 2016 21:47:13 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/LorentzVectorFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace HepMC {
  class FourVector;
}

//
// class declaration
//

class EmbeddingVertexCorrector : public edm::stream::EDProducer<> {
public:
  explicit EmbeddingVertexCorrector(const edm::ParameterSet &);
  ~EmbeddingVertexCorrector() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  // ----------member data ---------------------------
  edm::InputTag sourceLabel;
  edm::InputTag vertexPositionLabel;
};

//
// constructors and destructor
//
EmbeddingVertexCorrector::EmbeddingVertexCorrector(const edm::ParameterSet &iConfig) {
  produces<edm::HepMCProduct>();

  sourceLabel = iConfig.getParameter<edm::InputTag>("src");
  consumes<edm::HepMCProduct>(sourceLabel);
  vertexPositionLabel = edm::InputTag("externalLHEProducer", "vertexPosition");
  consumes<math::XYZTLorentzVectorD>(vertexPositionLabel);
}

EmbeddingVertexCorrector::~EmbeddingVertexCorrector() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EmbeddingVertexCorrector::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  // Retrieving generated Z to TauTau Event
  Handle<edm::HepMCProduct> InputGenEvent;
  iEvent.getByLabel(sourceLabel, InputGenEvent);
  HepMC::GenEvent *genevent = new HepMC::GenEvent(*InputGenEvent->GetEvent());
  std::unique_ptr<edm::HepMCProduct> CorrectedGenEvent(new edm::HepMCProduct(genevent));

  // Retrieving vertex position from input and creating vertex shift
  Handle<math::XYZTLorentzVectorD> vertex_position;
  iEvent.getByLabel(vertexPositionLabel, vertex_position);
  HepMC::FourVector vertex_shift(
      vertex_position.product()->x() * cm, vertex_position.product()->y() * cm, vertex_position.product()->z() * cm);

  // Apply vertex shift to all production vertices of the event
  CorrectedGenEvent->applyVtxGen(&vertex_shift);
  iEvent.put(std::move(CorrectedGenEvent));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void EmbeddingVertexCorrector::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  //  Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(EmbeddingVertexCorrector);
