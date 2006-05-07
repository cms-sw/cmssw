#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.1 2006/05/05 15:04:01 vanlaer Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Point3D.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

#include <vector>

class TrackingParticle;


class TrackingVertex {
 public:
  
  /// point in the space
  typedef math::XYZPoint Point;
  
  /// tracking particles
  typedef edm::Ref< std::vector<TrackingParticle> > TrackingParticleRef;
  typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
  typedef TrackingParticleContainer::iterator track_iterator;

  /// default constructor
  TrackingVertex();
  /// constructor from values
  TrackingVertex( const Point & );
  /// add a reference to a Track
  void add( const TrackingParticleRef & r );
  /// first iterator over tracks
  track_iterator tracks_begin() const ;
  /// last iterator over tracks
  track_iterator tracks_end() const ;
  
  /// position 
  const Point & position() const ;
  
 private:
  
  /// position
  Point position_;
  
  /// reference to tracks
  TrackingParticleContainer tracks_;
};



#endif
