#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.12 2006/06/30 19:21:04 ewv Exp $
 *
 */
 
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <vector>

using edm::SimVertexRef;
using edm::SimVertexRefVector;

class TrackingVertex {

 public:
  
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;
  typedef TrackingParticleContainer::iterator                  track_iterator;
  typedef        GenVertexRefVector::iterator                   genv_iterator;
  typedef        SimVertexRefVector::iterator                    g4v_iterator;
  
// Default constructor and constructor from values
  TrackingVertex();
  TrackingVertex(const HepLorentzVector &position, const bool inVolume, 
                 const int               source,   const int  crossing);

// Track and vertex iterators
  track_iterator     tracks_begin() const; // Ref's to TrackingParticle's
  track_iterator     tracks_end()   const; // associated with this vertex
  genv_iterator genVertices_begin() const; // Ref's to HepMC and Geant4
  genv_iterator genVertices_end()   const; // vertices associated with 
  g4v_iterator   g4Vertices_begin() const; // this vertex, respectively
  g4v_iterator   g4Vertices_end()   const; // ....

// Add references to TrackingParticle, Geant4, and HepMC vertices to correct containers
  void add(         const TrackingParticleRef&);
  void addG4Vertex( const SimVertexRef&       );
  void addGenVertex(const GenVertexRef&       );
  
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
  
  HepLorentzVector position_; // Vertex position and time
  bool inVolume_;             // Is it inside tracker volume?
  int  signalSource_;         // Is it signal or min-bias and in which crossing?
  
// References to G4 and generator vertices and tracks

  SimVertexRefVector  g4Vertices_;
  GenVertexRefVector genVertices_;
  TrackingParticleContainer tracks_;
};

#endif
