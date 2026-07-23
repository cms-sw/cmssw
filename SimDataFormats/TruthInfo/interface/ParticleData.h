// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimDataFormats_TruthInfo_interface_ParticleData_h
#define SimDataFormats_TruthInfo_interface_ParticleData_h

#include <cstdint>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/Checkpoint.h"

namespace truth {

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

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }
  };

}  // namespace truth

#endif
