#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;

// Constructors

TrackingVertex::TrackingVertex() : 
    position_(HepLorentzVector(0,0,0,0)), eId_(0) {}

TrackingVertex::TrackingVertex(const HepLorentzVector &p, const bool inVolume, 
                               const EncodedEventId eId) : 
    position_(p), inVolume_(inVolume), eId_(eId)  {}

/// add a reference to a Track
//void TrackingVertex::add( const TrackingParticleRef & r ) { tracks_.push_back( r ); }

/// add a reference to a vertex

void TrackingVertex::addG4Vertex(const SimVertexRef &ref) { 
  g4Vertices_.push_back(ref);
}

void TrackingVertex::addGenVertex(const GenVertexRef &ref){ 
  genVertices_.push_back(ref);
}
    
void TrackingVertex::addDaughterTrack(const TrackingParticleRef &ref){ 
  daughterTracks_.push_back(ref);
}
    
void TrackingVertex::addParentTrack(const TrackingParticleRef &ref){ 
  sourceTracks_.push_back(ref);
}
    
/// Iterators over tracks and vertices
//TrackingVertex::track_iterator TrackingVertex::tracks_begin()      const { return      tracks_.begin(); }
//TrackingVertex::track_iterator TrackingVertex::tracks_end()        const { return      tracks_.end();   }
TrackingVertex::genv_iterator  TrackingVertex::genVertices_begin() const { return genVertices_.begin(); }
TrackingVertex::genv_iterator  TrackingVertex::genVertices_end()   const { return genVertices_.end();   }
TrackingVertex::g4v_iterator   TrackingVertex::g4Vertices_begin()  const { return  g4Vertices_.begin(); }
TrackingVertex::g4v_iterator   TrackingVertex::g4Vertices_end()    const { return  g4Vertices_.end();   }

/// position 

const SimVertexRefVector TrackingVertex::g4Vertices() const {
  return  g4Vertices_;
};

const GenVertexRefVector TrackingVertex::genVertices() const {
  return  genVertices_;
};

//const TrackingParticleRefVector TrackingVertex::trackingParticles() const {
//  return  tracks_;
//};

