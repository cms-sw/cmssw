#ifndef Validation_TruthInfo_CandidateTruthMatching_h
#define Validation_TruthInfo_CandidateTruthMatching_h

// The pieces of the TICL candidate validation that do not depend on the framework: the particle classes a candidate
// can be, the projection of the truth graph on one level, the reconstruction ladder of one level member and the
// reco-driven classification of one candidate.

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace truth::candidate {

  enum class ParticleClass : uint8_t { Photon = 0, Electron, Muon, ChargedHadron, NeutralHadron, Other };
  inline constexpr std::size_t kNClasses = 6;
  inline constexpr std::array<const char*, kNClasses> kClassNames{
      "photon", "electron", "muon", "chargedHadron", "neutralHadron", "other"};
  // The identity bins of the reco-driven class matrix after the particle classes.
  inline constexpr std::size_t kIdentityMerged = kNClasses;
  inline constexpr std::size_t kIdentityPileup = kNClasses + 1;
  inline constexpr std::size_t kIdentityOutsideLevel = kNClasses + 2;
  inline constexpr std::size_t kIdentityNone = kNClasses + 3;
  inline constexpr std::size_t kNIdentityBins = kNClasses + 4;
  inline constexpr uint32_t kNoParticle = std::numeric_limits<uint32_t>::max();

  // The class of a truth particle. A pi0 is a photon: TICL reconstructs its photons.
  [[nodiscard]] ParticleClass truthClass(int32_t pdgId, double charge);
  // The class of a candidate PDG id. A PDG id that names no class, 0 included, is Other.
  [[nodiscard]] ParticleClass candidateClass(int pdgId);
  [[nodiscard]] inline bool isCharged(ParticleClass c) {
    return c == ParticleClass::Electron || c == ParticleClass::Muon || c == ParticleClass::ChargedHadron;
  }

  // The graph seen from one level, an antichain of members. A particle at or below a member maps to the position of
  // that member in the member list; a strict ancestor of a member maps to no member and is above the level.
  class LevelProjection {
  public:
    static constexpr uint32_t kNoMember = std::numeric_limits<uint32_t>::max();

    LevelProjection(Graph const& graph, std::span<const unsigned int> members);

    [[nodiscard]] uint32_t memberSlot(uint32_t particle) const {
      return particle < memberSlot_.size() ? memberSlot_[particle] : kNoMember;
    }
    [[nodiscard]] bool isAbove(uint32_t particle) const { return particle < above_.size() && above_[particle] != 0; }

  private:
    std::vector<uint32_t> memberSlot_;
    std::vector<uint8_t> above_;
  };

  // Whether one of the sorted particle ids is a strict ancestor of particle.
  [[nodiscard]] bool hasAncestorIn(Graph const& graph, uint32_t particle, std::span<const uint32_t> sortedIds);

  // A candidate whose calorimeter or track identity projects on one level member.
  struct MemberCandidate {
    // The fraction of the member's HGCAL energy the candidate shares, 0 for a candidate without a trackster.
    float sharedEnergyFraction = 0.f;
    ParticleClass candidateClass = ParticleClass::Other;
    bool hasTrack = false;
    // The track belongs to the same member.
    bool trackOnMember = false;
    double energy = 0.;
  };

  struct LadderConfig {
    double energyResponseMin;
    double energyResponseMax;
  };

  // The reconstruction ladder of one member: reconstructed, then the right track (a track of the member for a
  // charged member, no track for a neutral one), then the right class, then the energy response in the window.
  // Each step is read on the representative candidate: for a charged member a candidate with a track of the member,
  // else the candidate that shares most of the member energy.
  struct MemberOutcome {
    bool reconstructed = false;
    // Two or more candidates are on the member.
    bool multiple = false;
    bool trackCorrect = false;
    bool pidCorrect = false;
    bool energyCorrect = false;
    // Position of the representative candidate in the input span, -1 if none.
    int representative = -1;
    double energySum = 0.;
  };

  [[nodiscard]] MemberOutcome evaluateMember(ParticleClass memberClass,
                                             double trueEnergy,
                                             std::span<const MemberCandidate> candidates,
                                             LadderConfig const& config);

  // One entry of a reco-to-truth row: the root particle and the energy it shares with the candidate.
  struct RowMatch {
    uint32_t particle = kNoParticle;
    float sharedEnergy = 0.f;
  };

  // The share of the leading dominance-level member in the reference row, -1 without a member. A candidate without a
  // trackster has share 1 when its track particle projects on a member.
  [[nodiscard]] double leadingShare(LevelProjection const& dominance,
                                    std::span<const RowMatch> referenceRow,
                                    bool hasTrackster,
                                    uint32_t trackParticle);

  // The working-point-dependent inputs of one candidate.
  struct CandidateInput {
    ParticleClass candidateClass = ParticleClass::Other;
    bool hasTrackster = false;
    // The particle of row [0] of the working-point map and its score, kNoParticle without a match.
    uint32_t caloIdentity = kNoParticle;
    float caloScore = 1.f;
    bool hasTrack = false;
    // The particle of the track, kNoParticle when the track has none, and the purity of the track on it.
    uint32_t trackParticle = kNoParticle;
    float trackPurity = 0.f;
    // The value of leadingShare for this candidate.
    double leadingShare = -1.;
  };

  struct DominanceConfig {
    // The class of each dominance-level member, in member order.
    std::span<const ParticleClass> memberClass;
    double minLeadingTruthShare = 0.5;
  };

  // The reco-driven classification of one candidate. The identity is the calorimeter identity, or the track particle
  // for a candidate without a trackster.
  struct CandidateOutcome {
    uint32_t identity = kNoParticle;
    bool associated = false;
    bool calo = false;
    bool levelAssoc = false;
    bool merged = false;
    bool pileup = false;
    bool hasLevelCandidate = false;
    bool dominated = false;
    bool trackCalo = false;
    bool trackConsistent = false;
    bool pidCorrect = false;
    double purity = 0.;
    // A particle class, or kIdentityMerged, kIdentityPileup, kIdentityOutsideLevel or kIdentityNone.
    std::size_t identityBin = kIdentityNone;
  };

  [[nodiscard]] CandidateOutcome classifyCandidate(Graph const& graph,
                                                   LevelProjection const& dominance,
                                                   DominanceConfig const& config,
                                                   CandidateInput const& candidate);

}  // namespace truth::candidate

#endif
