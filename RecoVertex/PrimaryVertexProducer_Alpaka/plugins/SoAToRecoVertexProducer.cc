#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

/**
   * This plugin takes the SoA portableVertex and converts them to reco::Vertex, for usage within other workflows
   * - consuming set of reco::Tracks and portablevertex SoA
   * - produces a host reco::vertexCollection
 */
class SoAToRecoVertexProducer : public edm::stream::EDProducer<> {
public:
  SoAToRecoVertexProducer(edm::ParameterSet const& config)
      : portableVertexToken_(consumes(config.getParameter<edm::InputTag>("soaVertex"))),
        recoTrackToken_(consumes(config.getParameter<edm::InputTag>("srcTrack"))),
        recoVertexToken_(produces<reco::VertexCollection>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("soaVertex");
    desc.add<edm::InputTag>("srcTrack");

    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::EDGetTokenT<portablevertex::VertexHostCollection> portableVertexToken_;
  const edm::EDGetTokenT<reco::TrackCollection> recoTrackToken_;
  const edm::EDPutTokenT<reco::VertexCollection> recoVertexToken_;
};

void SoAToRecoVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Book inputs and space for outputs
  const portablevertex::VertexHostCollection& hostVertex = iEvent.get(portableVertexToken_);
  const portablevertex::VertexHostCollection::ConstView& hostVertexView = hostVertex.const_view();
  auto tracks =
      iEvent.getHandle(recoTrackToken_)
          .product();  // Note that we need reco::Tracks for building the track Reference vector inside the reco::Vertex

  // This is an annoying conversion as the vertex expects a transient track here, which is a dataformat which we otherwise bypass
  auto result = std::make_unique<reco::VertexCollection>();

  // Do the conversion back to reco::Vertex
  reco::VertexCollection& vColl = (*result);
  for (int iV = 0; iV < hostVertexView[0].nV(); iV++) {
    if (not(hostVertexView[iV].isGood()))
      continue;
    // Convert the SoA errors to a diagonal 3x3 matrix
    AlgebraicSymMatrix33 err;
    err[0][0] = hostVertexView[iV].errx();
    err[1][1] = hostVertexView[iV].erry();
    err[2][2] = hostVertexView[iV].errz();
    // Then we can actually create the vertex
    reco::Vertex newV(reco::Vertex::Point(hostVertexView[iV].x(), hostVertexView[iV].y(), hostVertexView[iV].z()),
                      err,
                      hostVertexView[iV].chi2(),
                      hostVertexView[iV].ndof(),
                      hostVertexView[iV].ntracks());
    // Finally, add references to the reco::Track used for building it
    for (int iT = 0; iT < hostVertexView[iV].ntracks(); iT++) {
      int new_itrack = hostVertexView[iV].track_id()[iT];
      reco::TrackRef ref(tracks, new_itrack);
      newV.add(ref, hostVertexView[iV].track_weight()[iT]);
    }
    vColl.push_back(newV);
  }
  // And finally put the collection in the event
  iEvent.put(std::move(result));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SoAToRecoVertexProducer);
