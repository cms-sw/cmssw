// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef SimDataFormats_TruthInfo_interface_VertexData_h
#define SimDataFormats_TruthInfo_interface_VertexData_h

#include <cstdint>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/InteractionId.h"

namespace truth {

  // Role of a logical vertex. Normal vertices are real GEN/SIM vertices.
  // Artificial source vertices summarize activity that was cut from a focused
  // selection but is kept for context/consistency:
  //   InitialState    - truncated production context of the selected roots: the
  //                     beam, the hard-scatter ancestry and the ISR that led to
  //                     the selection;
  //   UnderlyingEvent - stable final-state particles not in any selected
  //                     subgraph (underlying event, unrelated to the selection);
  //   BeamSideInput   - the input a kept vertex received from outside the selection.
  //                     It holds the dropped GEN parents of that vertex, which is
  //                     where a colour string spans the hard scatter and the beam
  //                     remnant, so that contribution stays visible.
  // Artificial vertices carry the genEvent/eventId of the activity they
  // summarize, so that overlaid pile-up graphs stay distinguishable. Interaction
  // is the per-interaction root that fans out (through connector particles) to
  // its InitialState and UnderlyingEvent sub-vertices, so the
  // whole interaction descends from one node: the signal is everything reachable
  // from the signal Interaction vertex, and each pile-up interaction gets its own.
  enum class VertexRole : uint8_t {
    Normal = 0,
    InitialState = 1,
    UnderlyingEvent = 2,
    Interaction = 3,
    BeamSideInput = 4
  };

  // Physical reason a vertex exists. A vertex with a SIM side takes it from the Geant4
  // creator-process subtype of the SimVertex (TruthGraph::nodeProcessType). A GEN-only
  // vertex takes it from its topology and from the generator status codes of the
  // particles that meet there, through genVertexReason(). Unknown for an artificial
  // vertex and for any pattern neither rule set covers.
  enum class VertexReason : uint8_t {
    Unknown = 0,
    Primary,            // no creator process (primary / hard-scatter)
    Decay,              // in-flight or radioactive decay
    Bremsstrahlung,     // brem photon emission
    Ionisation,         // delta-ray
    PairConversion,     // gamma -> e+e-
    Compton,            // Compton scattering
    PhotoElectric,      // photoelectric effect
    Annihilation,       // e+e- annihilation
    Rayleigh,           // Rayleigh scattering
    CoulombScattering,  // single Coulomb scattering
    HadronInelastic,    // nuclear / hadronic inelastic interaction
    HadronElastic,      // hadron elastic scattering
    NuclearCapture,     // (neutron) capture
    ChargeExchange,     // hadronic charge exchange
    HadronAtRest,       // hadron interaction at rest (e.g. pi- absorption)
    Other,              // a known-but-unmapped G4 subtype
    // GEN-only vertices. A generator decay reuses Decay above, which is the same
    // statement on either side.
    HardScatter,      // the hard interaction of the event
    ShowerBranching,  // a parton branching in the initial or final state shower
    Hadronization,    // partons or a string turn into hadrons
  };

  // Number of VertexRole values, for a bin per role.
  inline constexpr int kVertexRoleCount = static_cast<int>(VertexRole::BeamSideInput) + 1;

  [[nodiscard]] inline const char* vertexRoleName(VertexRole role) {
    switch (role) {
      case VertexRole::Normal:
        return "Normal";
      case VertexRole::InitialState:
        return "InitialState";
      case VertexRole::UnderlyingEvent:
        return "UnderlyingEvent";
      case VertexRole::Interaction:
        return "Interaction";
      case VertexRole::BeamSideInput:
        return "BeamSideInput";
    }
    return "Normal";
  }

  // Number of VertexReason values. A bin count or a sentinel derived from Other instead
  // would collide with the GEN values, which sit after it.
  inline constexpr int kVertexReasonCount = static_cast<int>(VertexReason::Hadronization) + 1;

  struct VertexData {
    // Optional provenance/debug back-references to the raw TruthGraph nodes.
    // -1 means "not available".
    int32_t genNode = -1;
    int32_t simNode = -1;

    // SIM event id when available, 0 otherwise.
    uint64_t eventId = 0;

    // GEN connected component id from the raw TruthGraph, -1 if not applicable.
    int32_t genEvent = -1;

    // VertexRole stored as its underlying type for dictionary simplicity.
    uint8_t role = static_cast<uint8_t>(VertexRole::Normal);

    // VertexReason, stored as its underlying type.
    uint8_t reason = static_cast<uint8_t>(VertexReason::Unknown);

    // Standalone payload.
    // Convention: "best available" position.
    // Prefer SIM if present, otherwise GEN, otherwise default-constructed.
    math::XYZTLorentzVectorD position;

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }

    [[nodiscard]] VertexRole vertexRole() const { return static_cast<VertexRole>(role); }
    [[nodiscard]] bool isArtificial() const { return vertexRole() != VertexRole::Normal; }

    [[nodiscard]] VertexReason vertexReason() const { return static_cast<VertexReason>(reason); }

    // The interaction this vertex belongs to, as on ParticleData.
    [[nodiscard]] int bunchCrossing() const { return bunchCrossingOf(eventId); }
    [[nodiscard]] int eventIndex() const { return eventIndexOf(eventId); }
    [[nodiscard]] bool isSignal() const { return isSignalEventId(eventId); }
    [[nodiscard]] bool isFromPileup() const { return !isSignal(); }
  };

  // Map a Geant4 process subtype (G4VProcess::GetProcessSubType(), as stored in
  // TruthGraph::nodeProcessType) to a VertexReason. Values from G4EmProcessSubType /
  // G4HadronicProcessType / G4DecayProcessType (Geant4 11.x). 0 = no creator process
  // = primary; unmapped-but-nonzero subtypes fall through to Other.
  [[nodiscard]] inline VertexReason reasonFromG4ProcessSubType(uint16_t subType) {
    switch (subType) {
      case 0:
        return VertexReason::Primary;
      case 1:
        return VertexReason::CoulombScattering;
      case 2:
        return VertexReason::Ionisation;
      case 3:
        return VertexReason::Bremsstrahlung;
      case 5:
        return VertexReason::Annihilation;
      case 11:
        return VertexReason::Rayleigh;
      case 12:
        return VertexReason::PhotoElectric;
      case 13:
        return VertexReason::Compton;
      case 14:
        return VertexReason::PairConversion;
      case 111:
        return VertexReason::HadronElastic;
      case 121:
        return VertexReason::HadronInelastic;
      case 131:
        return VertexReason::NuclearCapture;
      case 151:
        return VertexReason::HadronAtRest;
      case 161:
        return VertexReason::ChargeExchange;
      case 201:  // DECAY
      case 202:  // DECAY_WithSpin
      case 210:  // DECAY_Radioactive (== fRadioactiveDecay)
      case 211:  // DECAY_Unknown
      case 221:  // DECAY_MuAtom
        return VertexReason::Decay;
      default:
        return VertexReason::Other;
    }
  }

  [[nodiscard]] inline const char* vertexReasonName(VertexReason r) {
    switch (r) {
      case VertexReason::Unknown:
        return "Unknown";
      case VertexReason::Primary:
        return "Primary";
      case VertexReason::Decay:
        return "Decay";
      case VertexReason::Bremsstrahlung:
        return "Bremsstrahlung";
      case VertexReason::Ionisation:
        return "Ionisation";
      case VertexReason::PairConversion:
        return "PairConversion";
      case VertexReason::Compton:
        return "Compton";
      case VertexReason::PhotoElectric:
        return "PhotoElectric";
      case VertexReason::Annihilation:
        return "Annihilation";
      case VertexReason::Rayleigh:
        return "Rayleigh";
      case VertexReason::CoulombScattering:
        return "CoulombScattering";
      case VertexReason::HadronInelastic:
        return "HadronInelastic";
      case VertexReason::HadronElastic:
        return "HadronElastic";
      case VertexReason::NuclearCapture:
        return "NuclearCapture";
      case VertexReason::ChargeExchange:
        return "ChargeExchange";
      case VertexReason::HadronAtRest:
        return "HadronAtRest";
      case VertexReason::Other:
        return "Other";
      case VertexReason::HardScatter:
        return "HardScatter";
      case VertexReason::ShowerBranching:
        return "ShowerBranching";
      case VertexReason::Hadronization:
        return "Hadronization";
    }
    return "Unknown";
  }

}  // namespace truth

#endif
