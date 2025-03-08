#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_PORTABLETRACKSOAPRODUCER 0

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This:
   * - consumes set of reco::Tracks and reco::BeamSpot
   * - converts the reco::Tracks to a Alpaka-friendly dataformat portablevertex::TrackHostCollection
   * - puts the Alpaka dataformat in the device for later consumption
   */
  struct filterParameters {
    // Configurable filter parameters for the tracks
    double maxSignificance;
    double maxdxyError;
    double maxdzError;
    double minpAtIP;
    double maxetaAtIP;
    double maxchi2;
    int minpixelHits;
    int mintrackerHits;
    reco::TrackBase::TrackQuality trackQuality;
    double vertexSize;
    double d0CutOff;
  };

  class PortableTrackSoAProducer : public global::EDProducer<> {
  public:
    PortableTrackSoAProducer(edm::ParameterSet const& config)
        : theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
      theConfig = config;
      trackToken_ = consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("TrackLabel"));
      beamSpotToken_ = consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces();
      fParams = {
          .maxSignificance =
              config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Significance"),
          .maxdxyError =
              config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxD0Error"),
          .maxdzError = config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxDzError"),
          .minpAtIP = config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("minPt"),
          .maxetaAtIP = config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxEta"),
          .maxchi2 =
              config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("maxNormalizedChi2"),
          .minpixelHits =
              config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minPixelLayersWithHits"),
          .mintrackerHits =
              config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<int>("minSiliconLayersWithHits"),
          .trackQuality = reco::TrackBase::undefQuality,
          .vertexSize = config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("vertexSize"),
          .d0CutOff = config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<double>("d0CutOff")};
      std::string qualityClass =
          config.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<std::string>("trackQuality");
      if (qualityClass != "any" && qualityClass != "Any" && qualityClass != "ANY" && !(qualityClass.empty()))
        fParams.trackQuality = reco::TrackBase::qualityByName(qualityClass);
    }

    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // Get input collections from event
      auto tracks = iEvent.getHandle(trackToken_);
      auto beamSpotHandle = iEvent.getHandle(beamSpotToken_);
      reco::BeamSpot beamSpot;
      if (beamSpotHandle.isValid())
        beamSpot = *beamSpotHandle;
      int32_t tsize_ = tracks.product()->size();

      // Host collections
      portablevertex::TrackHostCollection hostTracks{tsize_, iEvent.queue()};
      auto& tview = hostTracks.view();

      // Fill Host collections with input, first initialize globals
      tview.totweight() = 0;
      tview.nT() = 0;

      // Build transient tracks
      const auto& theB = &iSetup.getData(theTTBToken);
      std::vector<reco::TransientTrack> t_tks;
      t_tks = (*theB).build(tracks, beamSpot);

      // We want to keep track of the original reco::Track index to later redo the conversion back to reco::Vertex
      std::vector<std::pair<int32_t, reco::TransientTrack>> sortedTracksPair;
      for (int32_t idx = 0; idx < tsize_; idx++) {
        sortedTracksPair.push_back(std::pair<int32_t, reco::TransientTrack>(idx, t_tks[idx]));
      }

      std::sort(sortedTracksPair.begin(),
                sortedTracksPair.end(),
                [](const std::pair<int32_t, reco::TransientTrack>& a,
                   const std::pair<int32_t, reco::TransientTrack>& b) -> bool {
                  return (a.second.stateAtBeamLine().trackStateAtPCA()).position().z() <
                         (b.second.stateAtBeamLine().trackStateAtPCA()).position().z();
                });

      int32_t nTrueTracks =
          0;  // This will keep track of how many we actually copy to device, only those that pass filter
      for (int32_t idx = 0; idx < tsize_; idx++) {
        // Fill up the the Track SoA, weight doubles up as an isGood flag, as we compute it only for good tracks
        double weight = convertTrack(tview[nTrueTracks],
                                     sortedTracksPair[idx].second,
                                     beamSpot,
                                     fParams,
                                     sortedTracksPair[idx].first,
                                     nTrueTracks);
        if (weight > 0) {
          nTrueTracks += 1;
          tview.nT() += 1;
          tview.totweight() += weight;
        }
      }
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_PORTABLETRACKSOAPRODUCER
      printf("[PortableTrackSoAProducer::produce()] From %i tracks, %i pass filters\n",
             (int32_t)tracks->size(),
             nTrueTracks);
#endif
      // Create device collections and copy into device
      portablevertex::TrackDeviceCollection deviceTracks{tsize_, iEvent.queue()};

      alpaka::memcpy(iEvent.queue(), deviceTracks.buffer(), hostTracks.buffer());

      // And put into the event
      iEvent.emplace(devicePutToken_, std::move(deviceTracks));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("TrackLabel");
      desc.add<edm::InputTag>("BeamSpotLabel");
      edm::ParameterSetDescription psd0;
      psd0.add<double>("maxNormalizedChi2", 10.0);
      psd0.add<double>("minPt", 0.0);
      psd0.add<std::string>("algorithm", "filter");
      psd0.add<double>("maxEta", 2.4);
      psd0.add<double>("maxD0Significance", 4.0);
      psd0.add<double>("maxD0Error", 1.0);
      psd0.add<double>("maxDzError", 1.0);
      psd0.add<std::string>("trackQuality", "any");
      psd0.add<int>("minPixelLayersWithHits", 2);
      psd0.add<int>("minSiliconLayersWithHits", 5);
      psd0.add<double>("vertexSize", 0.006);
      psd0.add<double>("d0CutOff", 0.10);
      desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<reco::TrackCollection> trackToken_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
    device::EDPutToken<portablevertex::TrackDeviceCollection> devicePutToken_;
    edm::ParameterSet theConfig;
    static double convertTrack(portablevertex::TrackHostCollection::View::element out,
                               const reco::TransientTrack in,
                               const reco::BeamSpot bs,
                               filterParameters fParams,
                               int32_t idx,
                               int32_t order);
    filterParameters fParams;
  };  //PortableTrackSoAProducer declaration

  double PortableTrackSoAProducer::convertTrack(portablevertex::TrackHostCollection::View::element out,
                                                const reco::TransientTrack in,
                                                const reco::BeamSpot bs,
                                                filterParameters fParams,
                                                int32_t idx,
                                                int32_t order) {
    bool isGood = false;
    double weight = -1;
    // First check if it passes filters
    if ((in.stateAtBeamLine().transverseImpactParameter().significance() < fParams.maxSignificance) &&
        (in.stateAtBeamLine().transverseImpactParameter().error() < fParams.maxdxyError) &&
        (in.track().dzError() < fParams.maxdzError) &&
        (in.impactPointState().globalMomentum().transverse() > fParams.minpAtIP) &&
        (std::fabs(in.impactPointState().globalMomentum().eta()) < fParams.maxetaAtIP) &&
        (in.normalizedChi2() < fParams.maxchi2) &&
        (in.hitPattern().pixelLayersWithMeasurement() >= fParams.minpixelHits) &&
        (in.hitPattern().trackerLayersWithMeasurement() >= fParams.mintrackerHits) &&
        (in.track().quality(fParams.trackQuality) || (fParams.trackQuality == reco::TrackBase::undefQuality)))
      isGood = true;
    if (isGood) {
      // Then define vertex-related stuff like weights
      weight = 1.;
      if (fParams.d0CutOff > 0) {
        // significance is measured in the transverse plane
        double significance = in.stateAtBeamLine().transverseImpactParameter().value() /
                              in.stateAtBeamLine().transverseImpactParameter().error();
        // weight is based on transverse displacement of the track
        weight = 1. / (1. + exp(std::pow(significance, 2) - std::pow(fParams.d0CutOff, 2)));
      }
      // Just fill up variables
      out.x() = in.stateAtBeamLine().trackStateAtPCA().position().x();
      out.y() = in.stateAtBeamLine().trackStateAtPCA().position().y();
      out.z() = in.stateAtBeamLine().trackStateAtPCA().position().z();
      out.px() = in.stateAtBeamLine().trackStateAtPCA().momentum().x();
      out.py() = in.stateAtBeamLine().trackStateAtPCA().momentum().y();
      out.pz() = in.stateAtBeamLine().trackStateAtPCA().momentum().z();
      out.weight() = weight;
      // The original index in the reco::Track collection so we can go back to it eventually
      out.tt_index() = idx;
      out.dz2() = std::pow(in.track().dzError(), 2);
      // Modified dz2 to account correlations and vertex size for clusterizer
      // dz^2 + (bs*pt)^2*pz^2/pt^2 + vertexSize^2
      double oneoverdz2 = (out.dz2()) +
                          ((std::pow(bs.BeamWidthX() * out.px(), 2)) + (std::pow(bs.BeamWidthY() * out.py(), 2))) *
                              std::pow(out.pz(), 2) /
                              std::pow(in.stateAtBeamLine().trackStateAtPCA().momentum().perp2(), 2) +
                          std::pow(fParams.vertexSize, 2);
      oneoverdz2 = 1. / oneoverdz2;
      out.oneoverdz2() = oneoverdz2;
      out.dxy2AtIP() = std::pow(in.track().dxyError(), 2);
      out.dxy2() = std::pow(in.stateAtBeamLine().transverseImpactParameter().error(), 2);
      out.order() = order;
      // All of these are initializers for the vertexing
      out.sum_Z() = 0;      // partition function sum
      out.kmin() = 0;       // minimum vertex identifier, will loop from kmin to kmax-1. At the start only one vertex
      out.kmax() = 1;       // maximum vertex identifier, will loop from kmin to kmax-1. At the start only one vertex
      out.aux1() = 0;       // for storing various things in between kernels
      out.aux2() = 0;       // for storing various things in between kernels
      out.isGood() = true;  // if we are here, we are to keep this track
    }
    return weight;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PortableTrackSoAProducer);
