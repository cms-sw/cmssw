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
  ~EmbeddingHltPixelVerticesProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginStream(edm::StreamID) override;
  virtual void produce(edm::Event &, const edm::EventSetup &) override;
  virtual void endStream() override;
  edm::InputTag vertexPositionLabel;
  // edm::InputTag generalTracks;

  // ----------member data ---------------------------
};

EmbeddingHltPixelVerticesProducer::EmbeddingHltPixelVerticesProducer(const edm::ParameterSet &iConfig) {
  vertexPositionLabel = edm::InputTag("externalLHEProducer", "vertexPosition");
  consumes<math::XYZTLorentzVectorD>(vertexPositionLabel);
  produces<reco::VertexCollection>();

  // generalTracks = iConfig.getParameter<edm::InputTag>("TrackLabel");
  // consumes<reco::TrackBaseRefVector>(generalTracks);
  // consumes<reco::TrackCollection>(generalTracks);
}

EmbeddingHltPixelVerticesProducer::~EmbeddingHltPixelVerticesProducer() {}

// ------------ method called to produce the data  ------------
void EmbeddingHltPixelVerticesProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  std::unique_ptr<reco::VertexCollection> embeddingVertex(new reco::VertexCollection);
  Handle<math::XYZTLorentzVectorD> vertex_position;
  iEvent.getByLabel(vertexPositionLabel, vertex_position);
  //  edm::LogPrint("") << "externalLHEProducer Vertex (" << vertex_position.product()->x() << ","  << vertex_position.product()->y() << "," <<  vertex_position.product()->z() << ")";
  math::XYZPoint genVertex =
      math::XYZPoint(vertex_position.product()->x(), vertex_position.product()->y(), vertex_position.product()->z());
  math::Error<3>::type Error;
  // additionally, get the general Tracks:
  // edm::Handle<reco::TrackCollection> tks;
  // iEvent.getByLabel(generalTracks, tks);
  // edm::LogPrint("") << "Loaded " << tks->size() << " tracks:";

  // edm::Handle<reco::TrackRef> tks_ref;
  // iEvent.getByLabel(generalTracks, tks_ref);
  // std::vector<edm::RefToBase<reco::Track> > tks_base_;
  // tks_base_.push_back(edm::RefToBase<reco::Track>(tks_ref));
  // reco::Vertex saveVertex = reco::Vertex(genVertex, Error);
  // Try to produce an nonfake Vertex
  // constructor for a valid vertex, with all data
  // Vertex( const Point &, const Error &, double chi2, double ndof, size_t size );
  // Need at least 5 ndof so the vertex Quality is considered good
  reco::Vertex saveVertex = reco::Vertex(genVertex, Error, 1.0, 6.0, 6);

  // for (auto track: *tks)
  //{
  // edm::LogPrint("") << track.vertex();
  // saveVertex.add(track, 0.5);
  //}
  // if (saveVertex.isFake()) edm::LogPrint("") << " The produced Vertex is fake";
  // else edm::LogPrint("") << " The produced Vertex is not fake";
  // edm::LogPrint("") << "Vertex Properties: " << saveVertex.isFake() << " / " << saveVertex.ndof() << " / " << abs(saveVertex.z()) << " / " << abs(saveVertex.position().Rho());
  // if (!saveVertex.isFake() && saveVertex.ndof() >= 4.0 && abs(saveVertex.z()) <= 24.0 && abs(saveVertex.position().Rho()) <= 2.0)
  // edm::LogPrint("") << "The Vertex is a goodOfflineVertex";
  embeddingVertex->push_back(saveVertex);
  // iEvent.put(std::move(embeddingVertex), "embeddingVertex");
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
