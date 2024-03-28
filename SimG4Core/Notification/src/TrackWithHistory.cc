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

// secondary particle
TrackWithHistory::TrackWithHistory(const G4Track* g4trk, const int pID) {
  trackID_ = g4trk->GetTrackID();
  pdgID_ = G4TrackToParticleID::particleID(g4trk);
  parentID_ = g4trk->GetParentID();
  mcTruthID_ = pID;
  auto mom = g4trk->GetMomentum();
  momentum_ = math::XYZVectorD(mom.x(), mom.y(), mom.z());
  totalEnergy_ = g4trk->GetTotalEnergy();
  auto pos = g4trk->GetPosition();
  vertexPosition_ = math::XYZVectorD(pos.x(), pos.y(), pos.z());
  time_ = g4trk->GetGlobalTime();
  auto p = g4trk->GetCreatorProcess();
  procType_ = (nullptr != p) ? p->GetProcessSubType() : 0;
  auto vgprimary = g4trk->GetDynamicParticle()->GetPrimaryParticle();
  int genID = 0;
  if (vgprimary != nullptr) {
    auto priminfo = static_cast<GenParticleInfo*>(vgprimary->GetUserInformation());
    if (nullptr != priminfo) {
      genID *= priminfo->id();
    }
  }
  // V.I. weight is computed in the same way as before
  // without usage of G4Track::GetWeight()
  weight_ = 10000 * genID;
}

// primary particle
TrackWithHistory::TrackWithHistory(const G4PrimaryParticle* ptr, int trackID, const math::XYZVectorD& pos, double time) {
  trackID_ = trackID;
  pdgID_ = G4TrackToParticleID::particleID(ptr, trackID_);
  parentID_ = 0;
  mcTruthID_ = trackID;
  auto mom = ptr->GetMomentum();
  momentum_ = math::XYZVectorD(mom.x(), mom.y(), mom.z());
  totalEnergy_ = ptr->GetTotalEnergy();
  vertexPosition_ = math::XYZVectorD(pos.x(), pos.y(), pos.z());
  time_ = time;
  storeTrack_ = true;
  int genID = 0;
  auto priminfo = static_cast<GenParticleInfo*>(ptr->GetUserInformation());
  if (nullptr != priminfo) {
    genID *= priminfo->id();
  }
  weight_ = 10000. * genID;
}
