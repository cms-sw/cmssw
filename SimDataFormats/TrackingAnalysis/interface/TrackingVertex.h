#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.8 2006/06/23 21:35:54 ewv Exp $
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
  
  /// point in the space
  typedef math::XYZPoint                                       Point;
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;
  
  /// tracking particles
  typedef edm::Ref< std::vector<TrackingParticle> >            TrackingParticleRef;
  typedef edm::RefVector< std::vector<TrackingParticle> >      TrackingParticleContainer;
  typedef TrackingParticleContainer::iterator                  track_iterator;
  
  /// default constructor
  TrackingVertex();
  /// constructor from values
  TrackingVertex( const HepLorentzVector & );
  /// add a reference to a Track
  void add( const TrackingParticleRef & r );
  /// first iterator over tracks
  track_iterator tracks_begin() const ;
  /// last iterator over tracks
  track_iterator tracks_end() const ;
  
  /// references to G4 and generator vertices
//  const  EmbdSimVertexRef &g4Vertex()  const { return g4Vertex_; }
//  GenVertexRef             genVertex() const { return genVertex_; }

  /// set references to G4 and generator vertices
//  void setG4Vertex( const EmbdSimVertexRef &r ) { g4Vertex_  = r; }
//  void setGenVertex(          GenVertexRef r )  { genVertex_ = r; }

  /// position 
  void addG4Vertex(const SimVertexRef &r);
  void addGenVertex(const GenVertexRef&);
  const HepLorentzVector & position() const ;
  const SimVertexRefVector    g4Vertices() const;
  const GenVertexRefVector        genVertices() const;
  const TrackingParticleRefVector trackingParticles() const;
//  const GenVertexRefVector genVertices() const;
  
 private:
  
  /// position
  HepLorentzVector position_;
  
  /// reference to tracks
  TrackingParticleContainer tracks_;

  /// references to G4 and generator vertices
  SimVertexRefVector g4Vertices_;
  GenVertexRefVector     genVertices_;
};



#endif
