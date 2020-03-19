#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "SimTracker/VertexAssociation/interface/VertexAssociatorByPositionAndTracks.h"
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

VertexAssociatorByPositionAndTracks::VertexAssociatorByPositionAndTracks(
    const edm::EDProductGetter *productGetter,
    double absZ,
    double sigmaZ,
    double maxRecoZ,
    double absT,
    double sigmaT,
    double maxRecoT,
    double sharedTrackFraction,
    const reco::RecoToSimCollection *trackRecoToSimAssociation,
    const reco::SimToRecoCollection *trackSimToRecoAssociation)
    : productGetter_(productGetter),
      absZ_(absZ),
      sigmaZ_(sigmaZ),
      maxRecoZ_(maxRecoZ),
      absT_(absT),
      sigmaT_(sigmaT),
      maxRecoT_(maxRecoT),
      sharedTrackFraction_(sharedTrackFraction),
      trackRecoToSimAssociation_(trackRecoToSimAssociation),
      trackSimToRecoAssociation_(trackSimToRecoAssociation) {}

VertexAssociatorByPositionAndTracks::VertexAssociatorByPositionAndTracks(
    const edm::EDProductGetter *productGetter,
    double absZ,
    double sigmaZ,
    double maxRecoZ,
    double sharedTrackFraction,
    const reco::RecoToSimCollection *trackRecoToSimAssociation,
    const reco::SimToRecoCollection *trackSimToRecoAssociation)
    : productGetter_(productGetter),
      absZ_(absZ),
      sigmaZ_(sigmaZ),
      maxRecoZ_(maxRecoZ),
      absT_(std::numeric_limits<double>::max()),
      sigmaT_(std::numeric_limits<double>::max()),
      maxRecoT_(std::numeric_limits<double>::max()),
      sharedTrackFraction_(sharedTrackFraction),
      trackRecoToSimAssociation_(trackRecoToSimAssociation),
      trackSimToRecoAssociation_(trackSimToRecoAssociation) {}

VertexAssociatorByPositionAndTracks::~VertexAssociatorByPositionAndTracks() {}

reco::VertexRecoToSimCollection VertexAssociatorByPositionAndTracks::associateRecoToSim(
    const edm::Handle<edm::View<reco::Vertex>> &vCH, const edm::Handle<TrackingVertexCollection> &tVCH) const {
  reco::VertexRecoToSimCollection ret(productGetter_);

  const edm::View<reco::Vertex> &recoVertices = *vCH;
  const TrackingVertexCollection &simVertices = *tVCH;

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::"
                                   "associateRecoToSim(): associating "
                                << recoVertices.size() << " reco::Vertices to" << simVertices.size()
                                << " TrackingVertices";

  // filter sim PVs
  std::vector<size_t> simPVindices;
  simPVindices.reserve(recoVertices.size());
  {
    int current_event = -1;
    for (size_t iSim = 0; iSim != simVertices.size(); ++iSim) {
      const TrackingVertex &simVertex = simVertices[iSim];

      // Associate only to primary vertices of the in-time pileup
      // events (BX=0, first vertex in each of the events)
      if (simVertex.eventId().bunchCrossing() != 0)
        continue;
      if (simVertex.eventId().event() != current_event) {
        current_event = simVertex.eventId().event();
        simPVindices.push_back(iSim);
      }
    }
  }

  for (size_t iReco = 0; iReco != recoVertices.size(); ++iReco) {
    const reco::Vertex &recoVertex = recoVertices[iReco];

    // skip fake vertices
    if (std::abs(recoVertex.z()) > maxRecoZ_ || recoVertex.isFake() || !recoVertex.isValid() || recoVertex.ndof() < 0.)
      continue;

    LogTrace("VertexAssociation") << " reco::Vertex at Z " << recoVertex.z();

    for (const size_t iSim : simPVindices) {
      const TrackingVertex &simVertex = simVertices[iSim];
      LogTrace("VertexAssociation") << "  Considering TrackingVertex at Z " << simVertex.position().z();

      //  recoVertex.t() == 0.  is a special value
      // need to change this to std::numeric_limits<double>::max() or something
      // more clear
      const bool useTiming = (absT_ != std::numeric_limits<double>::max() && recoVertex.t() != 0.);
      if (useTiming) {
        LogTrace("VertexAssociation") << " and T " << recoVertex.t() * CLHEP::second << std::endl;
      }

      const double tdiff = std::abs(recoVertex.t() - simVertex.position().t() * CLHEP::second);
      const double zdiff = std::abs(recoVertex.z() - simVertex.position().z());
      if (zdiff < absZ_ && zdiff / recoVertex.zError() < sigmaZ_ &&
          (!useTiming || (tdiff < absT_ && tdiff / recoVertex.tError() < sigmaT_))) {
        auto sharedTracks = calculateVertexSharedTracks(recoVertex, simVertex, *trackRecoToSimAssociation_);
        auto fraction = double(sharedTracks) / recoVertex.tracksSize();
        if (sharedTrackFraction_ < 0 || fraction > sharedTrackFraction_) {
          LogTrace("VertexAssociation") << "   Matched with significance " << zdiff / recoVertex.zError() << " "
                                        << tdiff / recoVertex.tError() << " shared tracks " << sharedTracks
                                        << " reco Tracks " << recoVertex.tracksSize() << " TrackingParticles "
                                        << simVertex.nDaughterTracks();

          ret.insert(reco::VertexBaseRef(vCH, iReco), std::make_pair(TrackingVertexRef(tVCH, iSim), sharedTracks));
        }
      }
    }
  }

  ret.post_insert();

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::associateRecoToSim(): finished";

  return ret;
}

reco::VertexSimToRecoCollection VertexAssociatorByPositionAndTracks::associateSimToReco(
    const edm::Handle<edm::View<reco::Vertex>> &vCH, const edm::Handle<TrackingVertexCollection> &tVCH) const {
  reco::VertexSimToRecoCollection ret(productGetter_);

  const edm::View<reco::Vertex> &recoVertices = *vCH;
  const TrackingVertexCollection &simVertices = *tVCH;

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::"
                                   "associateSimToReco(): associating "
                                << simVertices.size() << " TrackingVertices to " << recoVertices.size()
                                << " reco::Vertices";

  int current_event = -1;
  for (size_t iSim = 0; iSim != simVertices.size(); ++iSim) {
    const TrackingVertex &simVertex = simVertices[iSim];

    // Associate only primary vertices of the in-time pileup
    // events (BX=0, first vertex in each of the events)
    if (simVertex.eventId().bunchCrossing() != 0)
      continue;
    if (simVertex.eventId().event() != current_event) {
      current_event = simVertex.eventId().event();
    } else {
      continue;
    }

    LogTrace("VertexAssociation") << " TrackingVertex at Z " << simVertex.position().z();

    for (size_t iReco = 0; iReco != recoVertices.size(); ++iReco) {
      const reco::Vertex &recoVertex = recoVertices[iReco];

      // skip fake vertices
      if (std::abs(recoVertex.z()) > maxRecoZ_ || recoVertex.isFake() || !recoVertex.isValid() ||
          recoVertex.ndof() < 0.)
        continue;

      LogTrace("VertexAssociation") << "  Considering reco::Vertex at Z " << recoVertex.z();
      const bool useTiming = (absT_ != std::numeric_limits<double>::max() && recoVertex.t() != 0.);
      if (useTiming) {
        LogTrace("VertexAssociation") << " and T " << recoVertex.t() * CLHEP::second << std::endl;
      }

      const double tdiff = std::abs(recoVertex.t() - simVertex.position().t() * CLHEP::second);
      const double zdiff = std::abs(recoVertex.z() - simVertex.position().z());
      if (zdiff < absZ_ && zdiff / recoVertex.zError() < sigmaZ_ &&
          (!useTiming || (tdiff < absT_ && tdiff / recoVertex.tError() < sigmaT_))) {
        auto sharedTracks = calculateVertexSharedTracks(simVertex, recoVertex, *trackSimToRecoAssociation_);
        auto fraction = double(sharedTracks) / recoVertex.tracksSize();
        if (sharedTrackFraction_ < 0 || fraction > sharedTrackFraction_) {
          LogTrace("VertexAssociation") << "   Matched with significance " << zdiff / recoVertex.zError() << " "
                                        << tdiff / recoVertex.tError() << " shared tracks " << sharedTracks
                                        << " reco Tracks " << recoVertex.tracksSize() << " TrackingParticles "
                                        << simVertex.nDaughterTracks();

          ret.insert(TrackingVertexRef(tVCH, iSim), std::make_pair(reco::VertexBaseRef(vCH, iReco), sharedTracks));
        }
      }
    }
  }

  ret.post_insert();

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::associateSimToReco(): finished";

  return ret;
}
