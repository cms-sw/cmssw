#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.15 2006/08/22 16:12:45 ewv Exp $
 *
 */
 
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

using edm::SimVertexRef;
using edm::SimVertexRefVector;

class TrackingVertex {

 public:
  
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;

  typedef        GenVertexRefVector::iterator genv_iterator;
  typedef        SimVertexRefVector::iterator  g4v_iterator;
  typedef TrackingParticleRefVector::iterator   tp_iterator;
  
// Default constructor and constructor from values
  TrackingVertex();
  TrackingVertex(const HepLorentzVector &position, const bool inVolume, 
                 const EncodedEventId e = EncodedEventId(0));

// Setters
  void setEventId(EncodedEventId e) {eId_=e;};

// Track and vertex iterators
  genv_iterator genVertices_begin() const; // Ref's to HepMC and Geant4
  genv_iterator genVertices_end()   const; // vertices associated with 
  g4v_iterator   g4Vertices_begin() const; // this vertex, respectively
  g4v_iterator   g4Vertices_end()   const; // ....

  tp_iterator    daughterTracks_begin() const; // Ref's to daughter and source
  tp_iterator    daughterTracks_end()   const; // tracks associated with 
  tp_iterator      sourceTracks_begin() const; // this vertex, respectively
  tp_iterator      sourceTracks_end()   const; // ....

// Add references to TrackingParticles, Geant4, and HepMC vertices to containers
  void addG4Vertex(     const SimVertexRef&       );
  void addGenVertex(    const GenVertexRef&       );
  void addDaughterTrack(const TrackingParticleRef&);
  void addParentTrack(  const TrackingParticleRef&);
 
// Getters for RefVectors   
  const SimVertexRefVector            g4Vertices() const;
  const GenVertexRefVector           genVertices() const;
  const TrackingParticleRefVector   sourceTracks() const;
  const TrackingParticleRefVector daughterTracks() const;

// Getters for other info
  const HepLorentzVector& position() const { return position_; };
  const EncodedEventId     eventId() const { return eId_;      };
  const bool              inVolume() const { return inVolume_; }; 
  
 private:
  
  HepLorentzVector position_; // Vertex position and time
  bool             inVolume_; // Is it inside tracker volume?
  EncodedEventId   eId_;
  
// References to G4 and generator vertices and TrackingParticles

  SimVertexRefVector              g4Vertices_;
  GenVertexRefVector             genVertices_;
  TrackingParticleRefVector daughterTracks_;
  TrackingParticleRefVector   sourceTracks_;
};

#endif
