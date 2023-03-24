#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include "G4SystemOfUnits.hh"

class IdSort {
public:
  bool operator()(const SimTrack& a, const SimTrack& b) { return a.trackId() < b.trackId(); }
};

G4SimEvent::G4SimEvent() {
  g4vertices_.reserve(2000);
  g4tracks_.reserve(4000);
}

G4SimEvent::~G4SimEvent() { clear(); }

void G4SimEvent::clear() {
  for (auto& ptr : g4tracks_) {
    delete ptr;
  }
  g4tracks_.clear();
  for (auto& ptr : g4vertices_) {
    delete ptr;
  }
  g4vertices_.clear();
}

void G4SimEvent::load(edm::SimTrackContainer& c) const {
  const double invgev = 1.0 / CLHEP::GeV;
  for (auto& trk : g4tracks_) {
    int ip = trk->part();
    const math::XYZVectorD& mom = trk->momentum();
    math::XYZTLorentzVectorD p(mom.x() * invgev, mom.y() * invgev, mom.z() * invgev, trk->energy() * invgev);
    int iv = trk->ivert();
    int ig = trk->igenpart();
    int id = trk->id();
    // ip = particle ID as PDG
    // pp = 4-momentum in GeV
    // iv = corresponding G4SimVertex index
    // ig = corresponding GenParticle index
    SimTrack t = SimTrack(ip, p, iv, ig, trk->trackerSurfacePosition(), trk->trackerSurfaceMomentum());
    t.setTrackId(id);
    t.setEventId(EncodedEventId(0));
    t.setCrossedBoundaryVars(
        trk->crossedBoundary(), trk->getIDAtBoundary(), trk->getPositionAtBoundary(), trk->getMomentumAtBoundary());
    c.push_back(t);
  }
  std::stable_sort(c.begin(), c.end(), IdSort());
}

void G4SimEvent::load(edm::SimVertexContainer& c) const {
  const double invcm = 1.0 / CLHEP::cm;
  // index of the vertex is needed to make SimVertex object
  for (unsigned int i = 0; i < g4vertices_.size(); ++i) {
    G4SimVertex* vtx = g4vertices_[i];
    auto pos = vtx->vertexPosition();
    math::XYZVectorD v3(pos.x() * invcm, pos.y() * invcm, pos.z() * invcm);
    float t = vtx->vertexGlobalTime() / CLHEP::second;
    int iv = vtx->parentIndex();
    // v3 = position in cm
    // t  = global time in second
    // iv = index of the parent in the SimEvent SimTrack container (-1 if no parent)
    SimVertex v = SimVertex(v3, t, iv, i);
    v.setProcessType(vtx->processType());
    v.setEventId(EncodedEventId(0));
    c.push_back(v);
  }
}
