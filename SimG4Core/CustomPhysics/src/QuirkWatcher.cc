#include "SimG4Core/CustomPhysics/interface/QuirkWatcher.h"
#include "SimG4Core/CustomPhysics/interface/QuirkStringStore.h"
#include "SimG4Core/CustomPhysics/interface/Quirk.h"

#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4Exception.hh"
#include <CLHEP/Units/SystemOfUnits.h>

namespace {
  // kinetic energy given back to a stopped quirk; tiny, so no energy is pumped in
  constexpr G4double kRestEnergy = 1.0 * CLHEP::eV;
}  // namespace

QuirkWatcher::QuirkWatcher(G4double rMax, G4double zMax, G4bool keepStopped, G4double stopThreshold)
    : G4VProcess("QuirkWatcher"),
      m_rMax2((rMax > 0. && rMax < DBL_MAX) ? rMax * rMax : DBL_MAX),
      m_zMax((zMax > 0.) ? zMax : DBL_MAX),
      m_keepStopped(keepStopped),
      m_stopThreshold(stopThreshold) {
  enableAtRestDoIt = false;
  enableAlongStepDoIt = false;
}

G4double QuirkWatcher::PostStepGetPhysicalInteractionLength(const G4Track&, G4double, G4ForceCondition* condition) {
  *condition = StronglyForced;
  return DBL_MAX;
}

G4VParticleChange* QuirkWatcher::PostStepDoIt(const G4Track& track, const G4Step&) {
  if (!Quirk::isQuirk(track.GetParticleDefinition())) {
    G4Exception("QuirkWatcher::PostStepDoIt", "NonQuirk", FatalErrorInArgument, "QuirkWatcher run on non-quirk");
  }
  QuirkInfracolorForce& string =
      QuirkStringStore::instance().stringFor(track.GetParticleDefinition()->GetPDGEncoding());

  m_particleChange.Initialize(track);

  if (track.GetCurrentStepNumber() > 1 && !string.IsSourceInitialized()) {
    // orphan: clear the partner too, keep the hits recorded so far
    string.Clear();
    string.GetReactionForce()->Clear();
    G4Exception("QuirkWatcher::PostStepDoIt",
                "QuirkMissingPartner",
                JustWarning,
                "missing partner for quirk; killing orphan track");
    m_particleChange.ProposeTrackStatus(fStopAndKill);
    return &m_particleChange;
  }

  G4TrackStatus stat = track.GetTrackStatus();
  if (stat == fStopButAlive) {
    stat = fAlive;
  }
  // stopped by energy loss inside the world, partner still moving: the string pulls it on
  if (m_keepStopped && stat == fStopAndKill && track.GetKineticEnergy() <= 0. && track.GetNextVolume() != nullptr &&
      string.IsSourceAlive()) {
    const G4double partner = string.GetPartnerKineticEnergy();
    if (partner < 0. || partner > m_stopThreshold) {
      stat = fAlive;
      m_particleChange.ProposeEnergy(kRestEnergy);
    }
  }
  string.SetKineticEnergy(stat == fAlive ? std::max(track.GetKineticEnergy(), kRestEnergy) : track.GetKineticEnergy());
  // nothing to detect outside the envelope
  const G4ThreeVector& pos = track.GetPosition();
  if (pos.perp2() > m_rMax2 || std::abs(pos.z()) > m_zMax) {
    stat = fStopAndKill;
  }
  if (stat == fAlive || stat == fSuspend) {
    // no string left to absorb: pass control to the partner
    if (!string.HasNextStringVector()) {
      stat = string.IsSourceAlive() ? fSuspend : fStopAndKill;
    }
  }
  if (stat == fStopAndKill || stat == fKillTrackAndSecondaries) {
    string.TrackKilled();
  }

  m_particleChange.ProposeTrackStatus(stat);
  return &m_particleChange;
}
