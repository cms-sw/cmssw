#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4Core/Notification/interface/GenParticleInfoExtractor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"

#include <iostream>

G4Allocator<TrackWithHistory> TrackWithHistoryAllocator;

//#define DEBUG

G4TrackToParticleID * TrackWithHistory::theG4TrackToParticleID(0);

TrackWithHistory::TrackWithHistory(const G4Track * g4trk) : 
  trackID_(0),particleID_(0),parentID_(0),momentum_(math::XYZVectorD(0.,0.,0.)),
  totalEnergy_(0),vertexPosition_(math::XYZVectorD(0.,0.,0.)),globalTime_(0),
  localTime_(0),properTime_(0),creatorProcess_(0),weight_(0),
  storeTrack_(false),saved_(false) {
  
  if (theG4TrackToParticleID == 0) theG4TrackToParticleID = new G4TrackToParticleID;  
  if (g4trk!=0) {
    TrackInformationExtractor extractor;
    trackID_ = g4trk->GetTrackID();
    particleID_ = theG4TrackToParticleID->particleID(g4trk);
    parentID_ = g4trk->GetParentID();
    momentum_ = math::XYZVectorD(g4trk->GetMomentum().x(),g4trk->GetMomentum().y(),g4trk->GetMomentum().z());
    totalEnergy_ = g4trk->GetTotalEnergy();
    vertexPosition_ = math::XYZVectorD(g4trk->GetPosition().x(),g4trk->GetPosition().y(),g4trk->GetPosition().z());
    globalTime_  = g4trk->GetGlobalTime();
    localTime_  = g4trk->GetLocalTime();
    properTime_  = g4trk->GetProperTime();
    creatorProcess_ = g4trk->GetCreatorProcess();
    storeTrack_ = extractor(g4trk).storeTrack();
    saved_ = false;
    genParticleID_ = extractGenID( g4trk);
    // V.I. weight is computed in the same way as before
    // without usage of G4Track::GetWeight()
    weight_ = 10000*genParticleID_;
#ifdef DEBUG	
    LogDebug("TrackInformation") << " TrackWithHistory : created history for " << trackID_
				 << " with mother " << parentID_;
#endif
  }
}

void TrackWithHistory::checkAtEnd(const G4Track * gt) {

  math::XYZVectorD vposdir(gt->GetVertexPosition().x(),gt->GetVertexPosition().y(),gt->GetVertexPosition().z());
  math::XYZVectorD vmomdir(gt->GetVertexMomentumDirection().x(),gt->GetVertexMomentumDirection().y(),gt->GetVertexMomentumDirection().z());
  bool ok = true;
  double epsilon = 1.e-6;
  double eps2 = epsilon*epsilon;
  if ((vertexPosition_-vposdir).Mag2() > eps2)  {
    edm::LogWarning("TrackInformation") << "TrackWithHistory vertex position check failed" 
					<< "\nAt construction: " << vertexPosition_
					<< "\nAt end:          " << vposdir;
    ok = false;
  }
  math::XYZVectorD dirDiff = momentum_.Unit() - vmomdir;
  if (dirDiff.Mag2() > eps2 &&  momentum_.Unit().R() > eps2) {
    edm::LogWarning("TrackInformation") << "TrackWithHistory momentum direction check failed"
					<< "\nAt construction: " << momentum_.Unit() 
					<< "\nAt end:          " << vmomdir;
    ok = false;
  }
  if (!ok) throw SimG4Exception("TrackWithHistory::checkAtEnd failed");
}

int TrackWithHistory::extractGenID(const G4Track* gt) const {
  void * vgprimary = gt->GetDynamicParticle()->GetPrimaryParticle();
  if (vgprimary == 0) return -1;
  // replace old-style cast with appropriate new-style cast...
  G4PrimaryParticle* gprimary = (G4PrimaryParticle*) vgprimary;
  GenParticleInfoExtractor ext;
  return ext(gprimary).id();
}
