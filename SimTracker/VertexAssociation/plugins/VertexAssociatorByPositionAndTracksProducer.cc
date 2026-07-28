#include <limits>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

#include "SimTracker/VertexAssociation/interface/VertexAssociatorByPositionAndTracks.h"
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

/**
 * Constructs a VertexAssociatorByPositionAndTracks for the given vertex
 * collection type and puts the wrapped associator into the event, where it
 * can be consumed by downstream map-producing modules (VertexAssociatorEDProducer).
 *
 * Multiple track association maps may be provided via the `trackAssociations`
 * parameter to correctly handle vertices whose constituent tracks originate
 * from different collections (e.g. PF-based SVs mixing generalTracks and
 * gsfTracks). The maps are passed as a vector to the associator; each track
 * is looked up across all maps in turn.
 *
 * Registered plugins:
 *   VertexAssociatorByPositionAndTracksProducer
 *       for std::vector<reco::Vertex>
 *   VertexAssociatorByPositionAndTracksProducerCPC
 *       for std::vector<reco::VertexCompositePtrCandidate>
 */
template <typename VertexCollection>
class VertexAssociatorByPositionAndTracksProducerBase : public edm::global::EDProducer<> {
public:
  explicit VertexAssociatorByPositionAndTracksProducerBase(const edm::ParameterSet &);
  ~VertexAssociatorByPositionAndTracksProducerBase() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // Associator configuration — constants, safe to store as member data
  const double sigmaX_;
  const double sigmaY_;
  const double sigmaZ_;
  const double absZ_;
  const double maxRecoZ_;
  const double sigmaT_;
  const double absT_;
  const double maxRecoT_;
  const double sharedTrackFraction_;
  const bool filterSimVerticesForPVs_;
  const std::string weightMethod_;

  // One token pair per track collection. The i-th RecoToSim token and the
  // i-th SimToReco token must cover the same underlying track collection.
  std::vector<edm::EDGetTokenT<reco::RecoToSimCollection>> trackRecoToSimTokens_;
  std::vector<edm::EDGetTokenT<reco::SimToRecoCollection>> trackSimToRecoTokens_;
};

// =============================================================================
// Constructor
// =============================================================================

template <typename VertexCollection>
VertexAssociatorByPositionAndTracksProducerBase<VertexCollection>::VertexAssociatorByPositionAndTracksProducerBase(
    const edm::ParameterSet &config)
    : sigmaX_(config.getParameter<double>("sigmaX")),
      sigmaY_(config.getParameter<double>("sigmaY")),
      sigmaZ_(config.getParameter<double>("sigmaZ")),
      absZ_(config.getParameter<double>("absZ")),
      maxRecoZ_(config.getParameter<double>("maxRecoZ")),
      sigmaT_(config.getParameter<double>("sigmaT")),
      absT_(config.getParameter<double>("absT")),
      maxRecoT_(config.getParameter<double>("maxRecoT")),
      sharedTrackFraction_(config.getParameter<double>("sharedTrackFraction")),
      filterSimVerticesForPVs_(config.getParameter<bool>("filterSimVerticesForPVs")),
      weightMethod_(config.getParameter<std::string>("weightMethod")) {
  const auto trackAssociationTags = config.getParameter<std::vector<edm::InputTag>>("trackAssociations");

  if (trackAssociationTags.empty())
    throw cms::Exception("Configuration")
        << "VertexAssociatorByPositionAndTracksProducer: 'trackAssociations' must not be empty.";

  trackRecoToSimTokens_.reserve(trackAssociationTags.size());
  trackSimToRecoTokens_.reserve(trackAssociationTags.size());
  for (const auto &tag : trackAssociationTags) {
    trackRecoToSimTokens_.push_back(consumes<reco::RecoToSimCollection>(tag));
    trackSimToRecoTokens_.push_back(consumes<reco::SimToRecoCollection>(tag));
  }

  produces<reco::VertexToTrackingVertexAssociator<VertexCollection>>();
}

// =============================================================================
// produce
// =============================================================================

template <typename VertexCollection>
void VertexAssociatorByPositionAndTracksProducerBase<VertexCollection>::produce(edm::StreamID,
                                                                                edm::Event &iEvent,
                                                                                const edm::EventSetup &) const {
  // Build the association map vectors, one entry per configured track collection.
  RecoToSimCollectionVec recoToSimMaps;
  SimToRecoCollectionVec simToRecoMaps;
  recoToSimMaps.reserve(trackRecoToSimTokens_.size());
  simToRecoMaps.reserve(trackSimToRecoTokens_.size());

  bool anyInvalid = false;
  for (size_t i = 0; i < trackRecoToSimTokens_.size(); ++i) {
    edm::Handle<reco::RecoToSimCollection> recoToSimH;
    edm::Handle<reco::SimToRecoCollection> simToRecoH;
    iEvent.getByToken(trackRecoToSimTokens_[i], recoToSimH);
    iEvent.getByToken(trackSimToRecoTokens_[i], simToRecoH);

    if (!recoToSimH.isValid() || !simToRecoH.isValid()) {
      edm::LogWarning("VertexAssociatorByPositionAndTracksProducer")
          << "Track association collection at index " << i
          << " is not available in the event — skipping this collection.";
      anyInvalid = true;
      continue;
    }

    recoToSimMaps.push_back(recoToSimH.product());
    simToRecoMaps.push_back(simToRecoH.product());
  }

  if (recoToSimMaps.empty()) {
    edm::LogWarning("VertexAssociatorByPositionAndTracksProducer")
        << "No valid track association collections found — associator not produced.";
    return;
  }

  if (anyInvalid) {
    edm::LogWarning("VertexAssociatorByPositionAndTracksProducer")
        << "Some track association collections were missing; proceeding with " << recoToSimMaps.size() << " of "
        << trackRecoToSimTokens_.size() << " configured collections.";
  }

  // Construct the associator. The negative-value sentinel convention is handled
  // internally by the associator's getValueIfEnable(), so the full constructor
  // is always used regardless of whether timing is configured.
  auto impl = std::make_unique<VertexAssociatorByPositionAndTracks<VertexCollection>>(&(iEvent.productGetter()),
                                                                                      sigmaX_,
                                                                                      sigmaY_,
                                                                                      sigmaZ_,
                                                                                      absZ_,
                                                                                      maxRecoZ_,
                                                                                      sigmaT_,
                                                                                      absT_,
                                                                                      maxRecoT_,
                                                                                      sharedTrackFraction_,
                                                                                      std::move(recoToSimMaps),
                                                                                      std::move(simToRecoMaps),
                                                                                      weightMethod_,
                                                                                      filterSimVerticesForPVs_);

  iEvent.put(std::make_unique<reco::VertexToTrackingVertexAssociator<VertexCollection>>(std::move(impl)));
}

// =============================================================================
// fillDescriptions
// =============================================================================

template <typename VertexCollection>
void VertexAssociatorByPositionAndTracksProducerBase<VertexCollection>::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::vector<edm::InputTag>>("trackAssociations", {edm::InputTag("trackingParticleRecoTrackAsssociation")})
      ->setComment(
          "List of InputTags for track-TrackingParticle association maps "
          "(RecoToSimCollection and SimToRecoCollection). One entry suffices "
          "for vertices built from a single track collection (e.g. PVs from "
          "generalTracks). Provide multiple entries for PF-based SVs whose "
          "constituents may reference different track collections "
          "(e.g. generalTracks and gsfTracks).");

  desc.add<double>("sigmaX", -1.0)->setComment("Maximum dX / xError significance. Set negative to disable.");
  desc.add<double>("sigmaY", -1.0)->setComment("Maximum dY / yError significance. Set negative to disable.");
  desc.add<double>("sigmaZ", 3.0)->setComment("Maximum dZ / zError significance.");
  desc.add<double>("absZ", 0.1)->setComment("Maximum absolute dZ [cm].");
  desc.add<double>("maxRecoZ", 1000.0)->setComment("Maximum absolute Z of reco vertex to consider [cm].");
  desc.add<double>("sigmaT", -1.0)->setComment("Maximum dT / tError significance. Set negative to disable.");
  desc.add<double>("absT", -1.0)->setComment("Maximum absolute dT [s]. Set negative to disable.");
  desc.add<double>("maxRecoT", -1.0)
      ->setComment("Maximum absolute T of reco vertex to consider [s]. Set negative to disable.");
  desc.add<double>("sharedTrackFraction", -1.0)
      ->setComment(
          "Minimum shared-track fraction required for a match. "
          "Set to -1 to disable the track-content requirement (position-only).");
  desc.add<std::string>("weightMethod", "none")
      ->setComment(
          "Track weighting scheme for shared-track fraction: "
          "'none' (unweighted count), 'pt2', 'dzError', or 'nSharedTracks'.");
  desc.add<bool>("filterSimVerticesForPVs", true)
      ->setComment(
          "If true, only the first TrackingVertex per in-time pileup event (BX=0) "
          "is considered as a sim candidate. Set to true for PV, false for SV.");

  descriptions.addWithDefaultLabel(desc);
}

// =============================================================================
// Plugin registration
// =============================================================================

using VertexAssociatorByPositionAndTracksProducer =
    VertexAssociatorByPositionAndTracksProducerBase<std::vector<reco::Vertex>>;
DEFINE_FWK_MODULE(VertexAssociatorByPositionAndTracksProducer);

using VertexAssociatorByPositionAndTracksProducerCPC =
    VertexAssociatorByPositionAndTracksProducerBase<std::vector<reco::VertexCompositePtrCandidate>>;
DEFINE_FWK_MODULE(VertexAssociatorByPositionAndTracksProducerCPC);
