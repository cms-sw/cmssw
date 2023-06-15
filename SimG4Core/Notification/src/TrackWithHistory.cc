#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/GenParticleInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"
#include "G4DynamicParticle.hh"
#include "G4PrimaryParticle.hh"
#include "G4ThreeVector.hh"

#include <iostream>

G4ThreadLocal G4Allocator<TrackWithHistory>* fpTrackWithHistoryAllocator = nullptr;

//#define DEBUG

TrackWithHistory::TrackWithHistory(const G4Track* g4trk, int pID) {
  trackID_ = g4trk->GetTrackID();
  particleID_ = G4TrackToParticleID::particleID(g4trk);
  parentID_ = pID;
  auto mom = g4trk->GetMomentum();
  momentum_ = math::XYZVectorD(mom.x(), mom.y(), mom.z());
  totalEnergy_ = g4trk->GetTotalEnergy();
  auto pos = g4trk->GetPosition();
  vertexPosition_ = math::XYZVectorD(pos.x(), pos.y(), pos.z());
  globalTime_ = g4trk->GetGlobalTime();
  localTime_ = g4trk->GetLocalTime();
  properTime_ = g4trk->GetProperTime();
  creatorProcess_ = g4trk->GetCreatorProcess();
  TrackInformation* trkinfo = static_cast<TrackInformation*>(g4trk->GetUserInformation());
  storeTrack_ = trkinfo->storeTrack();
  auto vgprimary = g4trk->GetDynamicParticle()->GetPrimaryParticle();
  if (vgprimary != nullptr) {
    auto priminfo = static_cast<GenParticleInfo*>(vgprimary->GetUserInformation());
    if (nullptr != priminfo) {
      genParticleID_ = priminfo->id();
    }
  }
  // V.I. weight is computed in the same way as before
  // without usage of G4Track::GetWeight()
  weight_ = 10000 * genParticleID_;
#ifdef DEBUG
  LogDebug("TrackInformation") << " TrackWithHistory : created history for " << trackID_ << " with mother "
                               << parentID_;
#endif
}

TrackWithHistory::TrackWithHistory(const G4PrimaryParticle* ptr, int trackID, const math::XYZVectorD& pos, double time) {
  trackID_ = trackID;
  particleID_ = G4TrackToParticleID::particleID(ptr, trackID_);
  parentID_ = trackID;
  auto mom = ptr->GetMomentum();
  momentum_ = math::XYZVectorD(mom.x(), mom.y(), mom.z());
  totalEnergy_ = ptr->GetTotalEnergy();
  vertexPosition_ = math::XYZVectorD(pos.x(), pos.y(), pos.z());
  globalTime_ = localTime_ = properTime_ = time;
  creatorProcess_ = nullptr;
  storeTrack_ = true;
  auto priminfo = static_cast<GenParticleInfo*>(ptr->GetUserInformation());
  if (nullptr != priminfo) {
    genParticleID_ = priminfo->id();
  }
  weight_ = 10000. * genParticleID_;
#ifdef DEBUG
  LogDebug("TrackInformation") << " TrackWithHistory : created history for " << trackID_ << " with mother "
                               << parentID_;
#endif
}
