#ifndef TrackingVertex_h
#define TrackingVertex_h

/** \class TrackingVertex
 *  
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.13 2006/04/28 15:48:45 vanlaer Exp $
 *
 */
#include <Rtypes.h>
#include "DataFormats/Math/interface/Point3D.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include <vector>

class TrackingVertex {
  public:

    /// point in the space
    typedef math::XYZPoint Point;

    /// tracking particles
    typedef edm::Ref< std::vector<TrackingParticle> > TrackingParticleRef;
    typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;

    /// default constructor
    TrackingVertex() { }
    /// constructor from values
    TrackingVertex( const Point & );
    /// add a reference to a Track
    void add( const TrackingParticleRef & r ) { tracks_.push_back( r ); }
    /// first iterator over tracks
    track_iterator tracks_begin() const { return tracks_.begin(); }
    /// last iterator over tracks
    track_iterator tracks_end() const { return tracks_.end(); }

    /// position 
    const Point & position() const { return position_; }

  private:

    /// position
    Point position_;

    /// reference to tracks
    TrackingParticleContainer tracks_;
  };
  
}

#endif
