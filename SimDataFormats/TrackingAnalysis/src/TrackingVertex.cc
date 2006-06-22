#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

TrackingVertex::TrackingVertex( const TrackingVertex::Point & p) : position_(p) {}

TrackingVertex::TrackingVertex() : position_(Point(0,0,0)) {}

/// add a reference to a Track
void TrackingVertex::add( const TrackingParticleRef & r ) { tracks_.push_back( r ); }

/// add a reference to a vertex

void TrackingVertex::addG4Vertex(const EmbdSimVertexRef &ref) { 
  g4Vertices_.push_back(ref);
}

void TrackingVertex::addGenVertex(const GenVertexRef &ref){ 
  genVertices_.push_back(ref);
}
    
/// first iterator over tracks
TrackingVertex::track_iterator TrackingVertex::tracks_begin() const { return tracks_.begin(); }

/// last iterator over tracks
TrackingVertex::track_iterator TrackingVertex::tracks_end() const { return tracks_.end(); }

/// position 
const TrackingVertex::Point & TrackingVertex::position() const { return position_; }

const EmbdSimVertexRefVector TrackingVertex::g4Vertices() const {
  return  g4Vertices_;
};

const TrackingParticleRefVector TrackingVertex::trackingParticles() const {
  return  tracks_;
};

/*
const GenVertexRefVector TrackingVertex::genVertices() const {
  return genVertices_; 
};
*/
