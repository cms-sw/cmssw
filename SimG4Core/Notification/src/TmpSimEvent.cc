#include "SimG4Core/Notification/interface/TmpSimEvent.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SystemOfUnits.hh"

TmpSimEvent::TmpSimEvent() {
  g4vertices_.reserve(2000);
  g4tracks_.reserve(4000);
}

TmpSimEvent::~TmpSimEvent() { clear(); }

void TmpSimEvent::clear() {
  for (auto& ptr : g4tracks_) {
    delete ptr;
  }
  g4tracks_.clear();
  for (auto& ptr : g4vertices_) {
    delete ptr;
  }
  g4vertices_.clear();
}

void TmpSimEvent::load(edm::SimTrackContainer& cont) const {
  edm::LogVerbatim("SimG4CoreNotification") 
      << "TmpSimEvent::load Ntracks=" << g4tracks_.size();
  const double invgev = 1.0 / CLHEP::GeV;
  for (auto const & trk : g4tracks_) {
    int ip = trk->particleID();
    const math::XYZVectorD& mom = trk->momentum();
    math::XYZTLorentzVectorD p(mom.x() * invgev, mom.y() * invgev, mom.z() * invgev, trk->totalEnergy() * invgev);
    int iv = trk->vertexID();
    int ig = trk->mcTruthID();
    int id = trk->trackID();
    // id - track ID
    // ip - particle ID as PDG
    // p  - 4-momentum in GeV
    // iv - corresponding vertex index
    // ig - corresponding MC truth index
    SimTrack t = SimTrack(ip, p, iv, ig, trk->trackerSurfacePosition(), trk->trackerSurfaceMomentum());
    t.setTrackId(id);
    t.setEventId(EncodedEventId(0));
    t.setCrossedBoundaryVars(
        trk->crossedBoundary(), trk->getIDAtBoundary(), trk->getPositionAtBoundary(), trk->getMomentumAtBoundary());
    cont.push_back(t);
  }
}

void TmpSimEvent::load(edm::SimVertexContainer& cont) const {
  edm::LogVerbatim("SimG4CoreNotification") 
      << "TmpSimEvent::load Nvertices=" << g4vertices_.size();

  // index of the vertex is needed to make SimVertex object
  for (unsigned int i = 0; i < g4vertices_.size(); ++i) {
    TmpSimVertex* vtx = g4vertices_[i];
    float t = vtx->vertexGlobalTime() / CLHEP::second;
    int iv = vtx->parentIndex();
    // v3 = position in cm
    // t  = global time in second
    // iv = index of the parent in the SimEvent SimTrack container (-1 if no parent)
    SimVertex v = SimVertex(vtx->vertexPosition(), t, iv, i);
    v.setProcessType(vtx->processType());
    v.setEventId(EncodedEventId(0));
    cont.push_back(v);
  }
}
