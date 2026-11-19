#ifndef Validation_RecoVertex_SVValidationStructs_h
#define Validation_RecoVertex_SVValidationStructs_h

// Package:    Validation/RecoVertex
//
/**\struct SVValidationStructs Validation/RecoVertex/interface/SVValidationStructs.h

 Description: Internal structs representing simulated and reconstructed secondary
              vertices as used by the SecondaryVertexAnalyzer. These are lightweight
              analysis-time objects built from EDM types and association maps; they
              are never stored in the event.

 Original Author: Jan Schulz
*/

#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "Validation/RecoVertex/interface/SVEfficiencyEligibility.h"

// =============================================================================
// SimSecondaryVertex
//
// Represents a TrackingVertex that is considered as a secondary vertex truth
// candidate. Built from a TrackingVertexRef in SecondaryVertexAnalyzer::getSimSVs().
// =============================================================================

struct SimSecondaryVertex {
  SimSecondaryVertex(double x1, double y1, double z1)
      : x(x1),
        y(y1),
        z(z1),
        r(std::sqrt(x1 * x1 + y1 * y1)),
        decayLength(-1.),
        decayLengthXY(-1.),
        nCharged(0),
        nReconstructable(0),
        nMatchedRecoTracks(0),
        nMatchedRecoVertices(0),
        meanMatchedQuality(0.f),
        merged(false),
        motherPt(std::nullopt),
        motherPdgId(std::nullopt),
        isFromPileup(false) {}

  /// Reset all properties related to matching of reconstructed vertices/tracks
  void resetRecoDependencies() {
    nMatchedRecoVertices = 0;
    meanMatchedQuality = 0.0;
    matchedQualities.clear();
    nMatchedRecoTracks = 0;
    merged = false;
  }

  // Position
  double x, y, z;
  double r;  // transverse decay radius
  double dist(SimSecondaryVertex const &other) const {
    const double dx = x - other.x;
    const double dy = y - other.y;
    const double dz = z - other.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // Decay geometry — filled after PV association
  double decayLength;    // 3D decay length [cm]
  double decayLengthXY;  // 2D decay length [cm]

  // Kinematics
  math::XYZTLorentzVector chargedP4;
  double mass() const { return chargedP4.mass(); }
  double pt() const { return chargedP4.pt(); }
  double eta() const { return chargedP4.eta(); }
  double phi() const { return chargedP4.phi(); }

  // Daughter track multiplicity
  int nCharged;          // number of charged daughter TrackingParticles
  int nReconstructable;  // number of daughters with sufficient hits to be reconstructable

  // Matching to reco
  // matching quality depends on the vertex associator
  // (should be nSharedTracks for SVs, but could also be set to nSharedTracksFraction)
  int nMatchedRecoTracks;
  int nMatchedRecoVertices;
  float meanMatchedQuality;
  std::vector<float> matchedQualities;
  mutable bool merged;  // mutable so that it can be set over the RecoSecondaryVertex
  void setMerged() const { merged = true; }
  bool isMerged() const { return merged; }
  bool isMatched() const { return nMatchedRecoVertices > 0; }
  bool isReconstructable() const { return nMatchedRecoTracks >= 2; }

  // Generator-level information
  std::optional<double> motherPt;  // pt of the immediate decaying particle
  std::optional<int> motherPdgId;  // PDG ID of the immediate decaying particle
  bool isFromPileup;               // true if this vertex comes from a pileup interaction

  // Event identification
  EncodedEventId eventId;

  // Eligibility for efficiency calculation
  EfficiencyEligibility eligibility;

  // Reference to the underlying TrackingVertex
  TrackingVertexRef simVertex;
};

namespace detail {
  /// Decode an EfficiencyEligibility bitmask into a human-readable,
  /// comma-separated list of bundle names. Returns "none" if no bits are set.
  inline std::string eligibilityToString(EfficiencyEligibility mask) {
    std::string out;
    auto append = [&out](const char *name) {
      if (!out.empty())
        out += ",";
      out += name;
    };
    if ((mask & EfficiencyEligibility::kDecayLength) != EfficiencyEligibility::kNone)
      append("decayLength");
    if ((mask & EfficiencyEligibility::kNDaughters) != EfficiencyEligibility::kNone)
      append("nTracks");
    if ((mask & EfficiencyEligibility::kPt) != EfficiencyEligibility::kNone)
      append("pt");
    if ((mask & EfficiencyEligibility::kPdgId) != EfficiencyEligibility::kNone)
      append("pdgId");
    return out.empty() ? "none" : out;
  }
}  // namespace detail

/// Stream operator for debug printing of a SimSecondaryVertex.
/// Prints position, decay geometry, daughter multiplicity, mother PDG ID,
/// pileup flag, matching summary, and the efficiency-plot eligibility mask.
inline std::ostream &operator<<(std::ostream &os, const SimSecondaryVertex &sv) {
  os << "SimSecondaryVertex["
     << "pos=(" << sv.x << ", " << sv.y << ", " << sv.z << ") cm"
     << ", r=" << sv.r << " cm" << ", pt=" << sv.pt() << " GeV"
     << ", pt(mother)=" << sv.motherPt.value_or(0.) << " GeV" << ", phi=" << sv.phi() << " rad"
     << ", eta=" << sv.eta() << ", decayLength=" << sv.decayLength << " cm"
     << ", decayLengthXY=" << sv.decayLengthXY << " cm"
     << ", nCharged=" << sv.nCharged << ", nReconstructable=" << sv.nReconstructable
     << ", motherPdgId=" << sv.motherPdgId.value_or(0) << ", isFromPileup=" << (sv.isFromPileup ? "true" : "false")
     << ", nMatchedReco=" << sv.nMatchedRecoVertices << ", avgMatchQuality=" << sv.meanMatchedQuality
     << ", eligibleFor={" << detail::eligibilityToString(sv.eligibility) << "}"
     << "]";
  return os;
}

// =============================================================================
// RecoSecondaryVertex
//
// Represents a reconstructed secondary vertex. Templated on the underlying
// CMSSW type so the same struct can be used for both reco::Vertex (track-based)
// and reco::VertexCompositePtrCandidate (PF-based).
// =============================================================================

struct RecoSecondaryVertex {
  using Point = math::XYZPoint;
  using Error = math::Error<3>::type;

  // Bitmask flags — consistent with PrimaryVertexAnalyzer4PUSlimmed conventions
  enum VertexProperties {
    NONE = 0,
    MATCHED = 1,
    DUPLICATE = 2,
    FAKE = 4,
    MERGED = 8,
  };

  RecoSecondaryVertex(reco::Vertex const &vtx)
      : position(vtx.position()),
        positionCov(vtx.error()),
        chi2(vtx.chi2()),
        ndof(vtx.ndof()),
        nTracks(static_cast<int>(vtx.tracksSize())) {
    constexpr double kChargedPionMass = 0.13957039;  // GeV
    sumP4 = {0., 0., 0., 0.};
    for (auto iTrack = vtx.tracks_begin(); iTrack != vtx.tracks_end(); ++iTrack) {
      const double px = (*iTrack)->px();
      const double py = (*iTrack)->py();
      const double pz = (*iTrack)->pz();
      const double p2 = px * px + py * py + pz * pz;
      const double e = std::sqrt(p2 + kChargedPionMass * kChargedPionMass);
      sumP4 += math::XYZTLorentzVector(px, py, pz, e);
    }
  }

  RecoSecondaryVertex(reco::VertexCompositePtrCandidate const &vtx)
      : position(vtx.position()),
        positionCov(vtx.error()),
        sumP4(vtx.p4()),
        chi2(vtx.vertexChi2()),
        ndof(vtx.vertexNdof()),
        nTracks(vtx.numberOfDaughters()) {}

  // Position and kinematics
  Point position;
  double x() const { return position.x(); }
  double y() const { return position.y(); }
  double z() const { return position.z(); }
  double r() const { return position.rho(); }
  Error positionCov;
  double xError() const { return std::sqrt(positionCov(0, 0)); }
  double yError() const { return std::sqrt(positionCov(1, 1)); }
  double zError() const { return std::sqrt(positionCov(2, 2)); }
  math::XYZTLorentzVector sumP4;
  double mass() const { return sumP4.mass(); }
  double pt() const { return sumP4.pt(); }
  double eta() const { return sumP4.eta(); }
  double phi() const { return sumP4.phi(); }

  // Decay geometry
  Measurement1D decayLength3D;
  double decayLength() const { return decayLength3D.value(); }
  double decayLengthError() const { return decayLength3D.error(); }
  double decayLengthSignificance() const { return decayLength3D.significance(); }
  Measurement1D decayLength2D;
  double decayLengthXY() const { return decayLength2D.value(); }
  double decayLengthXYError() const { return decayLength2D.error(); }
  double decayLengthXYSignificance() const { return decayLength2D.significance(); }

  // Fit quality
  double chi2 = -1.;
  double ndof = -1.;
  double normalizedChi2() const { return (ndof > 0.) ? chi2 / ndof : -1.; }

  // Track multiplicity
  int nTracks = 0;

  // Matching to sim
  int nMatchedSimVertices = 0;
  std::vector<const SimSecondaryVertex *> simVertices;
  std::vector<float> matchedQualities;

  // Flagging the simVertices as merged
  void setSimMerged() const {
    for (auto const &sv : simVertices)
      sv->setMerged();
  }

  // Classification flags (bitmask of VertexProperties)
  int kind_of_vertex = NONE;

  // Optional fields — populated after MC truth matching
  std::optional<int> motherPdgId = std::nullopt;
  bool isFromPileup = false;

  // Reference to the underlying reco vertex:
  // reco::VertexBaseRef or VertexCompositePtrCandidateRef.
  std::optional<edm::RefToBase<reco::Vertex>> recoVertexRef = std::nullopt;
  std::optional<edm::RefToBase<reco::VertexCompositePtrCandidate>> recoVertexCPCRef = std::nullopt;

  template <typename VertexType>
  edm::RefToBase<VertexType> recoVertex() const;
};

/// Stream operator for debug printing of a SimSecondaryVertex.
/// Prints position, decay geometry, daughter multiplicity, mother PDG ID,
/// pileup flag, matching summary, and the efficiency-plot eligibility mask.
inline std::ostream &operator<<(std::ostream &os, const RecoSecondaryVertex &rv) {
  os << "RecoSecondaryVertex["
     << "pos=(" << rv.x() << ", " << rv.y() << ", " << rv.z() << ") cm"
     << ", r=" << rv.r() << " cm" << ", pt=" << rv.pt() << " GeV"
     << ", eta=" << rv.eta() << ", decayLength=" << rv.decayLength() << " cm"
     << ", decayLengthXY=" << rv.decayLengthXY() << " cm"
     << ", nTracks=" << rv.nTracks << ", nMatchedSim=" << rv.nMatchedSimVertices << "]";
  return os;
}

#endif  // Validation_RecoVertex_SVValidationStructs_h
