// -*- C++ -*-
//
// Package:    tautrigger/EmbeddingHltPixelVerticesProducer
// Class:      EmbeddingHltPixelVerticesProducer
//
/**\class EmbeddingHltPixelVerticesProducer EmbeddingHltPixelVerticesProducer.cc tautrigger/EmbeddingHltPixelVerticesProducer/plugins/EmbeddingHltPixelVerticesProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Sebastian Brommer
//         Created:  Thu, 02 Aug 2018 12:05:59 GMT
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

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Common/interface/RefToBase.h"

class EmbeddingHltPixelVerticesProducer : public edm::stream::EDProducer<> {
public:
  explicit EmbeddingHltPixelVerticesProducer(const edm::ParameterSet &);
  ~EmbeddingHltPixelVerticesProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;
  edm::InputTag vertexPositionLabel;

  // ----------member data ---------------------------
};

EmbeddingHltPixelVerticesProducer::EmbeddingHltPixelVerticesProducer(const edm::ParameterSet &iConfig) {
  vertexPositionLabel = edm::InputTag("externalLHEProducer", "vertexPosition");
  consumes<math::XYZTLorentzVectorD>(vertexPositionLabel);
  produces<reco::VertexCollection>();
}

EmbeddingHltPixelVerticesProducer::~EmbeddingHltPixelVerticesProducer() {}

// ------------ method called to produce the data  ------------
void EmbeddingHltPixelVerticesProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  std::unique_ptr<reco::VertexCollection> embeddingVertex(new reco::VertexCollection);
  Handle<math::XYZTLorentzVectorD> vertex_position;
  iEvent.getByLabel(vertexPositionLabel, vertex_position);
  math::XYZPoint genVertex =
      math::XYZPoint(vertex_position.product()->x(), vertex_position.product()->y(), vertex_position.product()->z());
  math::Error<3>::type Error;
  // Try to produce an nonfake Vertex
  // Need at least 5 ndof so the vertex Quality is considered good
  reco::Vertex saveVertex = reco::Vertex(genVertex, Error, 1.0, 6.0, 6);
  embeddingVertex->push_back(saveVertex);
  iEvent.put(std::move(embeddingVertex));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void EmbeddingHltPixelVerticesProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void EmbeddingHltPixelVerticesProducer::endStream() {}

void EmbeddingHltPixelVerticesProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  //  Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(EmbeddingHltPixelVerticesProducer);
