#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

#include "G4SystemOfUnits.hh"

class IdSort {
public:
  bool operator()(const SimTrack& a, const SimTrack& b) { return a.trackId() < b.trackId(); }
};

G4SimEvent::G4SimEvent()
    : hepMCEvent_(nullptr),
      weight_(0),
      collisionPoint_(math::XYZTLorentzVectorD(0., 0., 0., 0.)),
      nparam_(0),
      param_(0) {
  g4vertices_.reserve(2000);
  g4tracks_.reserve(4000);
}

G4SimEvent::~G4SimEvent() { clear(); }

void G4SimEvent::clear() {
  // per suggestion by Chris Jones, it's faster
  // that delete back() and pop_back()
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
  for (auto& trk : g4tracks_) {
    int ip = trk->part();
    math::XYZTLorentzVectorD p(
        trk->momentum().x() / GeV, trk->momentum().y() / GeV, trk->momentum().z() / GeV, trk->energy() / GeV);
    int iv = trk->ivert();
    int ig = trk->igenpart();
    int id = trk->id();
    math::XYZVectorD tkpos(trk->trackerSurfacePosition().x() / cm,
                           trk->trackerSurfacePosition().y() / cm,
                           trk->trackerSurfacePosition().z() / cm);
    math::XYZTLorentzVectorD tkmom(trk->trackerSurfaceMomentum().x() / GeV,
                                   trk->trackerSurfaceMomentum().y() / GeV,
                                   trk->trackerSurfaceMomentum().z() / GeV,
                                   trk->trackerSurfaceMomentum().e() / GeV);
    // ip = particle ID as PDG
    // pp = 4-momentum
    // iv = corresponding G4SimVertex index
    // ig = corresponding GenParticle index
    SimTrack t = SimTrack(ip, p, iv, ig, tkpos, tkmom);
    t.setTrackId(id);
    t.setEventId(EncodedEventId(0));
    if (trk->crossedBoundary())
      t.setCrossedBoundaryVars(
          trk->crossedBoundary(), trk->getIDAtBoundary(), trk->getPositionAtBoundary(), trk->getMomentumAtBoundary());
    c.push_back(t);
  }
  std::stable_sort(c.begin(), c.end(), IdSort());
}

void G4SimEvent::load(edm::SimVertexContainer& c) const {
  for (unsigned int i = 0; i < g4vertices_.size(); ++i) {
    G4SimVertex* vtx = g4vertices_[i];
    //
    // starting 1_1_0_pre3, SimVertex stores in cm !!!
    //
    math::XYZVectorD v3(vtx->vertexPosition().x() / cm, vtx->vertexPosition().y() / cm, vtx->vertexPosition().z() / cm);
    float t = vtx->vertexGlobalTime() / second;
    int iv = vtx->parentIndex();
    // vv = position
    // t  = global time
    // iv = index of the parent in the SimEvent SimTrack container (-1 if no parent)
    SimVertex v = SimVertex(v3, t, iv, i);
    v.setProcessType(vtx->processType());
    v.setEventId(EncodedEventId(0));
    c.push_back(v);
  }
}
