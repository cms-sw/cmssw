#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

/**
 * Fetches a VertexToTrackingVertexAssociator wrapper from the event and runs
 * both association directions, putting the resulting maps into the event.
 *
 * This module is intentionally associator-agnostic: it consumes any
 * VertexToTrackingVertexAssociator wrapper regardless of the underlying
 * implementation (single-map, multi-map, by-tracks, by-position, etc.).
 * All algorithm-specific configuration lives in the producer that creates
 * the wrapper (e.g. VertexAssociatorByPositionAndTracksProducer).
 *
 * This is the vertex-association analogue of TrackAssociatorEDProducer.
 *
 * Registered plugins:
 *   VertexAssociatorEDProducer
 *       for std::vector<reco::Vertex>
 *   VertexCompositePtrCandidateAssociatorEDProducer
 *       for std::vector<reco::VertexCompositePtrCandidate>
 */
template <typename VertexCollection>
class VertexAssociatorEDProducerBase : public edm::global::EDProducer<> {
public:
  using VertexType = typename VertexCollection::value_type;
  using AssociatorWrapper = reco::VertexToTrackingVertexAssociator<VertexCollection>;
  using SimToRecoCollection =
      typename reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::SimToRecoCollection;
  using RecoToSimCollection =
      typename reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::RecoToSimCollection;

  explicit VertexAssociatorEDProducerBase(const edm::ParameterSet &);
  ~VertexAssociatorEDProducerBase() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  const edm::EDGetTokenT<edm::View<VertexType>> recoVertexToken_;
  const edm::EDGetTokenT<TrackingVertexCollection> simVertexToken_;
  const edm::EDGetTokenT<AssociatorWrapper> associatorToken_;
};

// =============================================================================
// Constructor
// =============================================================================

template <typename VertexCollection>
VertexAssociatorEDProducerBase<VertexCollection>::VertexAssociatorEDProducerBase(const edm::ParameterSet &pset)
    : recoVertexToken_(consumes<edm::View<VertexType>>(pset.getParameter<edm::InputTag>("recoVertices"))),
      simVertexToken_(consumes<TrackingVertexCollection>(pset.getParameter<edm::InputTag>("simVertices"))),
      associatorToken_(consumes<AssociatorWrapper>(pset.getParameter<edm::InputTag>("associator"))) {
  produces<RecoToSimCollection>();
  produces<SimToRecoCollection>();
}

// =============================================================================
// produce
// =============================================================================

template <typename VertexCollection>
void VertexAssociatorEDProducerBase<VertexCollection>::produce(edm::StreamID,
                                                               edm::Event &iEvent,
                                                               const edm::EventSetup &) const {
  edm::Handle<edm::View<VertexType>> recoVertices;
  iEvent.getByToken(recoVertexToken_, recoVertices);

  edm::Handle<TrackingVertexCollection> simVertices;
  iEvent.getByToken(simVertexToken_, simVertices);

  edm::Handle<AssociatorWrapper> associator;
  iEvent.getByToken(associatorToken_, associator);

  LogTrace("VertexAssociation") << "VertexAssociatorEDProducer: associating " << recoVertices->size()
                                << " reco vertices to " << simVertices->size() << " TrackingVertices";

  auto recoToSim = associator->associateRecoToSim(recoVertices, simVertices);
  auto simToReco = associator->associateSimToReco(recoVertices, simVertices);

  iEvent.put(std::make_unique<RecoToSimCollection>(std::move(recoToSim)));
  iEvent.put(std::make_unique<SimToRecoCollection>(std::move(simToReco)));

  LogTrace("VertexAssociation") << "VertexAssociatorEDProducer: done\n";
}

// =============================================================================
// fillDescriptions
// =============================================================================

template <typename VertexCollection>
void VertexAssociatorEDProducerBase<VertexCollection>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recoVertices")->setComment("Reco vertex collection to associate.");
  desc.add<edm::InputTag>("simVertices")->setComment("TrackingVertex (sim truth) collection.");
  desc.add<edm::InputTag>("associator")
      ->setComment(
          "VertexToTrackingVertexAssociator wrapper produced by any vertex "
          "associator producer (e.g. VertexAssociatorByPositionAndTracksProducer). "
          "This module is agnostic to the underlying associator implementation.");

  descriptions.addWithDefaultLabel(desc);
}

// =============================================================================
// Plugin registration
// =============================================================================

using VertexAssociatorEDProducer = VertexAssociatorEDProducerBase<std::vector<reco::Vertex>>;
DEFINE_FWK_MODULE(VertexAssociatorEDProducer);

using VertexCompositePtrCandidateAssociatorEDProducer =
    VertexAssociatorEDProducerBase<std::vector<reco::VertexCompositePtrCandidate>>;
DEFINE_FWK_MODULE(VertexCompositePtrCandidateAssociatorEDProducer);
