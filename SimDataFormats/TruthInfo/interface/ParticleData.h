// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef SimDataFormats_TruthInfo_interface_ParticleData_h
#define SimDataFormats_TruthInfo_interface_ParticleData_h

#include <cstdint>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/Checkpoint.h"

namespace truth {

  // Membership of the graph levels, so a graph is self-describing wherever it is read.
  // Only levels re-derivable from what the graph stores belong here; that is the rule
  // for adding one. BranchSelector-dependent sets stay per-event products downstream,
  // so a threshold change never forces a re-production. Every bit can be recomputed
  // with levelAntichain() (Signal and ReconstructableFromSignal from the seed lists
  // recorded on the Graph), which is what the dumper audit and LevelFlags_t check.
  enum class LevelFlag : uint32_t {
    StableLegsFromUpstream = 1u << 0,
    HardProcess = 1u << 1,
    StableDecayProducts = 1u << 2,
    CaloBoundary = 1u << 3,
    // The resonance the preset's seed species name: the most upstream matching GEN
    // particles of the signal interaction. Empty recorded seeds mean no Signal bits.
    Signal = 1u << 4,
    // First reconstructable decay products of the signal: the walk from each Signal
    // root stops at the graph's reconstructablePdgIds (a pi0 is one object, not two
    // photons) or at a generator-stable particle, passes through intermediates the
    // detector cannot see as objects, and drops invisible species.
    ReconstructableFromSignal = 1u << 5,
    // Stable legs of the artificial UnderlyingEvent vertex, the spectator counterpart
    // of StableLegsFromUpstream. Empty without a selection preset, not wrong.
    UnderlyingEvent = 1u << 6,
    // One root per parton-initiated jet: the hard-process legs that are partons, each
    // standing for its descendant subgraph; no clustering, flavour = the parton's own
    // PDG id. A subset of HardProcess, so a top contributes its b, never itself.
    PartonJets = 1u << 7,
    // The hadron of each heavy-flavour chain that DECAYS WEAKLY, which is the one CMS
    // ghost association names: "the generated b and c hadrons that do not have b and c
    // hadrons as daughters respectively" (CMS-BTV-16-002). A B* radiating to a B counts
    // once, as the B. Beauty and charm are separate levels because a B decays to a D, and
    // a combined level would silently drop every charm member.
    BHadrons = 1u << 8,
    CHadrons = 1u << 9,
    // The visible final state of the whole GENERATOR record: the reconstructableFromSignal
    // walk seeded from every GEN root instead of from the Signal roots, so a pi0 is one
    // object inside a QCD jet, the underlying event and every pileup interaction, where no
    // resonance exists to seed from.
    ReconstructableFinalState = 1u << 10,
    // Hadronically decaying taus, one per physical tau: the last tau of each radiative
    // chain, with a GEN decay record and no electron and no muon among its decay children.
    // The object tau identification measures efficiency against.
    VisibleTau = 1u << 11,
  };

  // What a particle IS, mirroring VertexRole on the vertex side. Absence of a GEN and a
  // SIM back-reference does NOT identify a synthetic particle: connectors have neither,
  // and so would anything else artificial, so the kind has to be stated rather than
  // inferred. Guessing it from empty fields silently conflated the two.
  enum class ParticleRole : uint8_t {
    // A generator or Geant4 particle.
    Normal = 0,
    // Artificial: produced at an Interaction vertex and decaying at the Upstream or
    // UnderlyingEvent sub-vertex, so those descend from one interaction root.
    Connector = 1,
    // Artificial: stands in for a resonance the generator never wrote, so the signal
    // level is answerable on a non-resonant sample. Its momentum is an ACCOUNTING sum
    // over the hard-process legs and is not a generator quantity.
    SignalStandIn = 2,
  };

  struct ParticleData {
    // Optional provenance/debug back-references to the raw TruthGraph nodes.
    // -1 means "not available".
    int32_t genNode = -1;
    int32_t simNode = -1;

    // Merged metadata.
    int32_t pdgId = 0;
    int16_t status = 0;

    // Packed reco::GenStatusFlags bitfield, when available.
    // 0 means "not available" or "no flags set".
    uint16_t statusFlags = 0;

    // SIM event id when available, 0 otherwise.
    uint64_t eventId = 0;

    // GEN connected component id from the raw TruthGraph, -1 if not applicable.
    int32_t genEvent = -1;

    // Bitwise OR of the LevelFlag values this particle belongs to. fillLevelFlags owns
    // every bit except Signal, which the selection post-processing sets from the seed
    // species and which travels on the particle so it survives the graph rewrite.
    // Occupies the alignment hole between genEvent and momentum, so sizeof stays 96
    // (asserted in LevelFlags_t). Zero is ambiguous, "belongs to no level" or "written
    // before this member existed", so a reader re-derives with levelAntichain().
    uint32_t levelFlags = 0;

    // Standalone payload.
    // Nominal physics four-momentum.
    // For GEN+SIM particles, this is the GEN four-momentum.
    // For SIM-only particles, this is the SimTrack four-momentum.
    math::XYZTLorentzVectorD momentum;

    // Optional trajectory checkpoints.
    std::vector<Checkpoint> checkpoints;

    // True for SIM particles that Geant4 flagged as back-scattered (albedo): the
    // track crossed the Tracker<->CALO boundary inward. From SimTrack::isFromBack-
    // Scattering(); always false for GEN-only particles.
    bool backscattered = false;

    // Real particle, connector, or synthetic stand-in, stored as its underlying type
    // for dictionary simplicity as VertexData::role is. Sits in the tail padding after
    // backscattered, so carrying it keeps sizeof(ParticleData) at 96.
    uint8_t role = static_cast<uint8_t>(ParticleRole::Normal);

    [[nodiscard]] ParticleRole particleRole() const { return static_cast<ParticleRole>(role); }

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }

    // True for anything the graph invented. Never read the momentum of such a particle
    // as a generator quantity.
    [[nodiscard]] bool isSynthetic() const { return particleRole() != ParticleRole::Normal; }

    [[nodiscard]] bool isAtLevel(LevelFlag flag) const { return (levelFlags & static_cast<uint32_t>(flag)) != 0; }
    void setLevel(LevelFlag flag) { levelFlags |= static_cast<uint32_t>(flag); }
  };

}  // namespace truth

#endif
