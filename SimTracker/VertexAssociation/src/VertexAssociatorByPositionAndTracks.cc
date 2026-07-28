#include <limits>

#include <CLHEP/Units/SystemOfUnits.h>

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimTracker/VertexAssociation/interface/VertexAssociatorByPositionAndTracks.h"
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

// =============================================================================
// Constructors
// =============================================================================

namespace {
  void parseWeightMethod(const std::string &weightMethod,
                         bool &useWeightPtSum2,
                         bool &useWeightDzErr,
                         bool &useNSharedTracks) {
    if (weightMethod == "pt2")
      useWeightPtSum2 = true;
    else if (weightMethod == "dzError")
      useWeightDzErr = true;
    else if (weightMethod == "nSharedTracks")
      useNSharedTracks = true;
    else if (weightMethod != "none")
      throw cms::Exception("Configuration")
          << "VertexAssociatorByPositionAndTracks: Invalid weightMethod '" << weightMethod
          << "' (should be 'none', 'pt2', 'dzError' or 'nSharedTracks')";
  }
}  // namespace

template <typename VertexCollection>
VertexAssociatorByPositionAndTracks<VertexCollection>::VertexAssociatorByPositionAndTracks(
    const edm::EDProductGetter *productGetter,
    double sigmaX,
    double sigmaY,
    double sigmaZ,
    double absZ,
    double maxRecoZ,
    double sigmaT,
    double absT,
    double maxRecoT,
    double sharedTrackFraction,
    RecoToSimCollectionVec trackRecoToSimAssociations,
    SimToRecoCollectionVec trackSimToRecoAssociations,
    const std::string &weightMethod,
    bool filterSimVerticesForPVs)
    : productGetter_(productGetter),
      sigmaX_(getValueIfEnable(sigmaX)),
      sigmaY_(getValueIfEnable(sigmaY)),
      sigmaZ_(getValueIfEnable(sigmaZ)),
      absZ_(getValueIfEnable(absZ)),
      maxRecoZ_(getValueIfEnable(maxRecoZ)),
      sigmaT_(getValueIfEnable(sigmaT)),
      absT_(getValueIfEnable(absT)),
      maxRecoT_(getValueIfEnable(maxRecoT)),
      sharedTrackFraction_(sharedTrackFraction),
      trackRecoToSimAssociations_(std::move(trackRecoToSimAssociations)),
      trackSimToRecoAssociations_(std::move(trackSimToRecoAssociations)),
      useWeightPtSum2_(false),
      useWeightDzErr_(false),
      useNSharedTracks_(false),
      filterSimVerticesForPVs_(filterSimVerticesForPVs) {
  parseWeightMethod(weightMethod, useWeightPtSum2_, useWeightDzErr_, useNSharedTracks_);
}

template <typename VertexCollection>
VertexAssociatorByPositionAndTracks<VertexCollection>::VertexAssociatorByPositionAndTracks(
    const edm::EDProductGetter *productGetter,
    double sigmaX,
    double sigmaY,
    double sigmaZ,
    double absZ,
    double maxRecoZ,
    double sharedTrackFraction,
    RecoToSimCollectionVec trackRecoToSimAssociations,
    SimToRecoCollectionVec trackSimToRecoAssociations,
    const std::string &weightMethod,
    bool filterSimVerticesForPVs)
    : productGetter_(productGetter),
      sigmaX_(getValueIfEnable(sigmaX)),
      sigmaY_(getValueIfEnable(sigmaY)),
      sigmaZ_(getValueIfEnable(sigmaZ)),
      absZ_(getValueIfEnable(absZ)),
      maxRecoZ_(getValueIfEnable(maxRecoZ)),
      sigmaT_(kCheckDisabled),
      absT_(kCheckDisabled),
      maxRecoT_(kCheckDisabled),
      sharedTrackFraction_(sharedTrackFraction),
      trackRecoToSimAssociations_(std::move(trackRecoToSimAssociations)),
      trackSimToRecoAssociations_(std::move(trackSimToRecoAssociations)),
      useWeightPtSum2_(false),
      useWeightDzErr_(false),
      useNSharedTracks_(false),
      filterSimVerticesForPVs_(filterSimVerticesForPVs) {
  parseWeightMethod(weightMethod, useWeightPtSum2_, useWeightDzErr_, useNSharedTracks_);
}

// =============================================================================
// Private helpers — specialised per VertexType
// =============================================================================

// -----------------------------------------------------------------------------
// reco::Vertex specialisations
// -----------------------------------------------------------------------------

template <>
bool VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::isRecoVertexInvalid(const reco::Vertex &vtx) const {
  return vtx.isFake() || !vtx.isValid() || vtx.ndof() < 0.;
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexX(const reco::Vertex &vtx) const {
  return vtx.x();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexY(const reco::Vertex &vtx) const {
  return vtx.y();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexZ(const reco::Vertex &vtx) const {
  return vtx.z();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexXError(const reco::Vertex &vtx) const {
  return vtx.xError();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexYError(const reco::Vertex &vtx) const {
  return vtx.yError();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexZError(const reco::Vertex &vtx) const {
  return vtx.zError();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexT(const reco::Vertex &vtx) const {
  return vtx.t();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>::recoVertexTError(const reco::Vertex &vtx) const {
  return vtx.tError();
}

// -----------------------------------------------------------------------------
// reco::VertexCompositePtrCandidate specialisations
// -----------------------------------------------------------------------------

template <>
bool VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::isRecoVertexInvalid(
    const reco::VertexCompositePtrCandidate &vtx) const {
  // VertexCompositePtrCandidate has no isFake()/isValid() interface.
  // Reject vertices with no daughters as a basic sanity check.
  return vtx.numberOfDaughters() == 0;
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexX(
    const reco::VertexCompositePtrCandidate &vtx) const {
  return vtx.vx();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexY(
    const reco::VertexCompositePtrCandidate &vtx) const {
  return vtx.vy();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexZ(
    const reco::VertexCompositePtrCandidate &vtx) const {
  return vtx.vz();
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexXError(
    const reco::VertexCompositePtrCandidate &vtx) const {
  // vertexCovariance(i,j): indices 0=x, 1=y, 2=z
  return std::sqrt(vtx.vertexCovariance(0, 0));
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexYError(
    const reco::VertexCompositePtrCandidate &vtx) const {
  return std::sqrt(vtx.vertexCovariance(1, 1));
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexZError(
    const reco::VertexCompositePtrCandidate &vtx) const {
  return std::sqrt(vtx.vertexCovariance(2, 2));
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexT(
    const reco::VertexCompositePtrCandidate & /*vtx*/) const {
  // Timing is not available for VertexCompositePtrCandidate.
  // Returning 0. disables the timing cut (consistent with reco::Vertex
  // convention where t() == 0. signals no timing information).
  return 0.;
}

template <>
double VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>::recoVertexTError(
    const reco::VertexCompositePtrCandidate & /*vtx*/) const {
  return std::numeric_limits<double>::max();
}

// -----------------------------------------------------------------------------
// shared implementation for both Vertex and VertexCompositePtrCandidate
// -----------------------------------------------------------------------------

template <typename VertexCollection>
float VertexAssociatorByPositionAndTracks<VertexCollection>::sharedTrackFractionForVertex(
    const VertexType &recoVertex, const TrackingVertex &simVertex) const {
  auto sharedTracksAndFraction = calculateVertexSharedTracks(recoVertex, simVertex, trackRecoToSimAssociations_);
  if (useWeightPtSum2_)
    return sharedTracksAndFraction.sharedPt2Fraction_;
  if (useWeightDzErr_)
    return sharedTracksAndFraction.sharedDzErrFraction_;
  if (useNSharedTracks_)
    return sharedTracksAndFraction.nSharedTracks_;
  return sharedTracksAndFraction.sharedTracksFraction_;
}

template <typename VertexCollection>
auto VertexAssociatorByPositionAndTracks<VertexCollection>::makeVertexRef(
    const edm::Handle<edm::View<VertexType>> &handle, size_t index) const {
  return handle->refAt(index);
}

// =============================================================================
// Association loops
// =============================================================================

template <typename VertexCollection>
typename VertexAssociatorByPositionAndTracks<VertexCollection>::RecoToSimCollection
VertexAssociatorByPositionAndTracks<VertexCollection>::associateRecoToSim(
    const edm::Handle<edm::View<VertexType>> &vCH, const edm::Handle<TrackingVertexCollection> &tVCH) const {
  RecoToSimCollection ret(productGetter_);

  const edm::View<VertexType> &recoVertices = *vCH;
  const TrackingVertexCollection &simVertices = *tVCH;

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::"
                                   "associateRecoToSim(): associating "
                                << recoVertices.size() << " reco vertices to " << simVertices.size()
                                << " TrackingVertices";

  const bool useSigmaX = sigmaX_ != kCheckDisabled;
  const bool useSigmaY = sigmaY_ != kCheckDisabled;

  // Build the list of sim vertex indices to consider.
  // For PV association (filterSimVerticesForPVs_=true) only the first
  // TrackingVertex per in-time pileup event is kept.
  // For SV association (filterSimVerticesForPVs_=false) all in-time
  // TrackingVertices are considered.
  std::vector<size_t> simIndicesToConsider;
  simIndicesToConsider.reserve(simVertices.size());
  {
    int current_event = -1;
    for (size_t iSim = 0; iSim != simVertices.size(); ++iSim) {
      const TrackingVertex &simVertex = simVertices[iSim];
      if (simVertex.eventId().bunchCrossing() != 0)
        continue;
      if (filterSimVerticesForPVs_) {
        if (simVertex.eventId().event() != current_event) {
          current_event = simVertex.eventId().event();
          simIndicesToConsider.push_back(iSim);
        }
      } else {
        simIndicesToConsider.push_back(iSim);
      }
    }
  }

  for (size_t iReco = 0; iReco != recoVertices.size(); ++iReco) {
    const VertexType &recoVertex = recoVertices[iReco];

    if (isRecoVertexInvalid(recoVertex))
      continue;

    const double recoZ = recoVertexZ(recoVertex);
    if (std::abs(recoZ) > maxRecoZ_)
      continue;

    LogTrace("VertexAssociation") << " RecoVertex at X,Y,Z " << recoVertexX(recoVertex) << ","
                                  << recoVertexY(recoVertex) << "," << recoZ;

    const double recoT = recoVertexT(recoVertex);
    const bool useTiming = (absT_ != kCheckDisabled && recoT != 0.);

    for (const size_t iSim : simIndicesToConsider) {
      const TrackingVertex &simVertex = simVertices[iSim];

      const double xdiff = useSigmaX ? std::abs(recoVertexX(recoVertex) - simVertex.position().x()) : 0.;
      const double ydiff = useSigmaY ? std::abs(recoVertexY(recoVertex) - simVertex.position().y()) : 0.;
      const double zdiff = std::abs(recoZ - simVertex.position().z());
      const double tdiff = useTiming ? std::abs(recoT - simVertex.position().t() * CLHEP::second) : 0.;

      if (useSigmaX && (xdiff / recoVertexXError(recoVertex) >= sigmaX_))
        continue;
      if (useSigmaY && (ydiff / recoVertexYError(recoVertex) >= sigmaY_))
        continue;
      if (zdiff >= absZ_ || zdiff / recoVertexZError(recoVertex) >= sigmaZ_)
        continue;
      if (useTiming && (tdiff >= absT_ || tdiff / recoVertexTError(recoVertex) >= sigmaT_))
        continue;

      const float fraction = sharedTrackFractionForVertex(recoVertex, simVertex);
      if (sharedTrackFraction_ >= 0 && fraction < sharedTrackFraction_)
        continue;

      LogTrace("VertexAssociation") << "   Matched at X,Y,Z=" << simVertex.position().x() << ","
                                    << simVertex.position().y() << "," << simVertex.position().z()
                                    << ": dZ significance " << zdiff / recoVertexZError(recoVertex)
                                    << " shared track fraction " << fraction;

      ret.insert(makeVertexRef(vCH, iReco), std::make_pair(TrackingVertexRef(tVCH, iSim), fraction));
    }
  }

  ret.post_insert();

  LogTrace("VertexAssociation") << "VertexAssociatorByPositionAndTracks::associateRecoToSim(): finished";

  return ret;
}

template <typename VertexCollection>
typename VertexAssociatorByPositionAndTracks<VertexCollection>::SimToRecoCollection
VertexAssociatorByPositionAndTracks<VertexCollection>::associateSimToReco(
    const edm::Handle<edm::View<VertexType>> &vCH, const edm::Handle<TrackingVertexCollection> &tVCH) const {
  SimToRecoCollection ret(productGetter_);

  const edm::View<VertexType> &recoVertices = *vCH;
  const TrackingVertexCollection &simVertices = *tVCH;

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::"
                                   "associateSimToReco(): associating "
                                << simVertices.size() << " TrackingVertices to " << recoVertices.size()
                                << " reco vertices";

  const bool useSigmaX = sigmaX_ != kCheckDisabled;
  const bool useSigmaY = sigmaY_ != kCheckDisabled;

  int current_event = -1;
  for (size_t iSim = 0; iSim != simVertices.size(); ++iSim) {
    const TrackingVertex &simVertex = simVertices[iSim];

    if (simVertex.eventId().bunchCrossing() != 0)
      continue;

    if (filterSimVerticesForPVs_) {
      if (simVertex.eventId().event() != current_event) {
        current_event = simVertex.eventId().event();
      } else {
        continue;
      }
    }

    LogTrace("VertexAssociation") << " TrackingVertex at X,Y,Z " << simVertex.position().x() << ","
                                  << simVertex.position().y() << "," << simVertex.position().z();

    for (size_t iReco = 0; iReco != recoVertices.size(); ++iReco) {
      const VertexType &recoVertex = recoVertices[iReco];

      if (isRecoVertexInvalid(recoVertex))
        continue;

      const double recoZ = recoVertexZ(recoVertex);
      if (std::abs(recoZ) > maxRecoZ_)
        continue;

      const double recoT = recoVertexT(recoVertex);
      const bool useTiming = (absT_ != kCheckDisabled && recoT != 0.);

      const double xdiff = useSigmaX ? std::abs(recoVertexX(recoVertex) - simVertex.position().x()) : 0.;
      const double ydiff = useSigmaY ? std::abs(recoVertexY(recoVertex) - simVertex.position().y()) : 0.;
      const double zdiff = std::abs(recoZ - simVertex.position().z());
      const double tdiff = useTiming ? std::abs(recoT - simVertex.position().t() * CLHEP::second) : 0.;

      if (useSigmaX && (xdiff / recoVertexXError(recoVertex) >= sigmaX_))
        continue;
      if (useSigmaY && (ydiff / recoVertexYError(recoVertex) >= sigmaY_))
        continue;
      if (zdiff >= absZ_ || zdiff / recoVertexZError(recoVertex) >= sigmaZ_)
        continue;
      if (useTiming && (tdiff >= absT_ || tdiff / recoVertexTError(recoVertex) >= sigmaT_))
        continue;

      const float fraction = sharedTrackFractionForVertex(recoVertex, simVertex);
      if (sharedTrackFraction_ >= 0 && fraction < sharedTrackFraction_)
        continue;

      LogTrace("VertexAssociation") << "   Matched: dZ significance " << zdiff / recoVertexZError(recoVertex)
                                    << " shared track fraction " << fraction;

      ret.insert(TrackingVertexRef(tVCH, iSim), std::make_pair(makeVertexRef(vCH, iReco), fraction));
    }
  }

  ret.post_insert();

  LogDebug("VertexAssociation") << "VertexAssociatorByPositionAndTracks::associateSimToReco(): finished";

  return ret;
}

// =============================================================================
// Explicit instantiations
// =============================================================================

template class VertexAssociatorByPositionAndTracks<std::vector<reco::Vertex>>;
template class VertexAssociatorByPositionAndTracks<std::vector<reco::VertexCompositePtrCandidate>>;
