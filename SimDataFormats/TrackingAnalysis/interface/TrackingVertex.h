#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.9 2006/06/27 16:53:24 ewv Exp $
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
  
  /// default constructor
  TrackingVertex();
  /// constructor from values
  TrackingVertex(const HepLorentzVector &, const bool inVolume, 
                 const int source,         const int  crossing);
  /// first iterator over tracks
  track_iterator tracks_begin() const ;
  /// last iterator over tracks
  track_iterator tracks_end() const ;
  

// Add references to reference containers
  void add( const TrackingParticleRef & r );
  void addG4Vertex(const SimVertexRef &r);
  void addGenVertex(const GenVertexRef&);
  
// Getters for RefVectors   
  const SimVertexRefVector        g4Vertices() const;
  const GenVertexRefVector        genVertices() const;
  const TrackingParticleRefVector trackingParticles() const;

  const HepLorentzVector & position() const ;
  
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
