#ifndef SimG4Core_G4SimEvent_H
#define SimG4Core_G4SimEvent_H

#include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimG4Core/Application/interface/G4SimVertex.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "HepMC/GenEvent.h"
// #include "CLHEP/Vector/LorentzVector.h"

#include <vector>

class G4SimEvent
{
public:
    G4SimEvent();
    virtual ~G4SimEvent();
    void load(edm::SimTrackContainer & c) const;
    void load(edm::SimVertexContainer & c)  const;
    unsigned int nTracks() const { return g4tracks.size(); }
    unsigned int nVertices() const { return g4vertices.size(); }
    unsigned int nGenParts() const { return hepMCEvent->particles_size(); }
    void hepEvent(const HepMC::GenEvent * r) { hepMCEvent = r; }
    const HepMC::GenEvent * hepEvent() const { return hepMCEvent; }
    void weight(float w) { weight_ = w; }
    float weight() const { return weight_; }
    void collisionPoint( const math::XYZTLorentzVectorD& v ) { collisionPoint_ = v ; }
    const math::XYZTLorentzVectorD& collisionPoint() const { return collisionPoint_; }
    void nparam(int n) { nparam_ = n; }
    const int nparam() const { return nparam_; }
    void param(const std::vector<float>& p) { param_ = p; }
    const std::vector<float> & param() const { return param_; }
    void add(G4SimTrack * t) { g4tracks.push_back(t); }
    void add(G4SimVertex * v) { g4vertices.push_back(v); }
    const G4SimTrack & g4track(int i) const { return *g4tracks[i-1]; }
    const G4SimVertex & g4vertex(int i) const { return *g4vertices[i-1]; }
protected:
    const HepMC::GenEvent * hepMCEvent;  
    float weight_;
    math::XYZTLorentzVectorD collisionPoint_;
    int nparam_;
    std::vector<float> param_;
    std::vector<G4SimTrack *> g4tracks;
    std::vector<G4SimVertex *> g4vertices;
};

#endif
