#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;

// Constructors

TrackingVertex::TrackingVertex() : 
    position_(HepLorentzVector(0,0,0,0)), eId_(0) {}

TrackingVertex::TrackingVertex(const HepLorentzVector &p, const bool inVolume, 
                               const EncodedEventId eId) : 
    position_(p), inVolume_(inVolume), eId_(eId)  {}

// Add a reference to vertex vectors

void TrackingVertex::addG4Vertex(const SimVertexRef &ref) { 
  g4Vertices_.push_back(ref);
}

void TrackingVertex::addGenVertex(const GenVertexRef &ref){ 
  genVertices_.push_back(ref);
}
    
// Add a reference to track vectors

void TrackingVertex::addDaughterTrack(const TrackingParticleRef &ref){ 
  daughterTracks_.push_back(ref);
}
    
void TrackingVertex::addParentTrack(const TrackingParticleRef &ref){ 
  sourceTracks_.push_back(ref);
}
    
// Iterators over vertices and tracks

TrackingVertex::genv_iterator TrackingVertex::genVertices_begin() const { return genVertices_.begin(); }
TrackingVertex::genv_iterator TrackingVertex::genVertices_end()   const { return genVertices_.end();   }
TrackingVertex::g4v_iterator  TrackingVertex::g4Vertices_begin()  const { return  g4Vertices_.begin(); }
TrackingVertex::g4v_iterator  TrackingVertex::g4Vertices_end()    const { return  g4Vertices_.end();   }

TrackingVertex::tp_iterator TrackingVertex::daughterTracks_begin() const { return daughterTracks_.begin(); }
TrackingVertex::tp_iterator TrackingVertex::daughterTracks_end()   const { return daughterTracks_.end();   }
TrackingVertex::tp_iterator TrackingVertex::sourceTracks_begin()   const { return sourceTracks_.begin();   }
TrackingVertex::tp_iterator TrackingVertex::sourceTracks_end()     const { return sourceTracks_.end();     }

// Accessors for whole vectors

const SimVertexRefVector        TrackingVertex::g4Vertices()     const { return  g4Vertices_;     };
const GenVertexRefVector        TrackingVertex::genVertices()    const { return  genVertices_;    };
const TrackingParticleRefVector TrackingVertex::sourceTracks()   const { return  sourceTracks_;   };
const TrackingParticleRefVector TrackingVertex::daughterTracks() const { return  daughterTracks_; };
