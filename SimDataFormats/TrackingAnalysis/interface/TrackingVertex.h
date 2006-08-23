#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.10 2006/06/28 17:15:29 ewv Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <vector>

using edm::SimVertexRef;
using edm::SimVertexRefVector;

namespace HepMC {
  class GenVertex;
}

class TrackingParticle;


class TrackingVertex {
 public:
  
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;
  typedef TrackingParticleContainer::iterator                  track_iterator;
  
// Default constructor
  TrackingVertex();
// Constructor from values
  TrackingVertex(const HepLorentzVector&, const bool inVolume, 
                 const int source,        const int  crossing);

// Track iterators
  track_iterator tracks_begin() const ;
  track_iterator tracks_end()   const ;

// Add references to reference containers
  void add( const TrackingParticleRef & r );
  void addG4Vertex(const SimVertexRef &r);
  void addGenVertex(const GenVertexRef&);
  
// Getters for RefVectors   
  const SimVertexRefVector         g4Vertices()       const;
  const GenVertexRefVector        genVertices()       const;
  const TrackingParticleRefVector trackingParticles() const;

// Getters for other info
  const HepLorentzVector& position() const; // Position and time
  const bool              isSignal() const; // Is from signal process
  const int               crossing() const; // Crossing number (-n ... +n)
  const int               source()   const; // Signal source
  const bool              inVolume() const; // Inside tracking volume
  
 private:
  
  /// position
  HepLorentzVector position_;
  
  /// reference to tracks
  TrackingParticleContainer tracks_;

  /// references to G4 and generator vertices
  SimVertexRefVector  g4Vertices_;
  GenVertexRefVector genVertices_;
  bool inVolume_;          // Is it inside tracker volume?
  int  signalSource_;      // Is it signal or min-bias and in which crossing?
};



#endif
