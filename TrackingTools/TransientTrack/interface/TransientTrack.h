#ifndef TrackReco_TransientTrack_h
#define TrackReco_TransientTrack_h
//
// Definition of Transient Track class for 
// reconstruction posterior to track reconstruction (vertexing, b-tagging...)
//

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

namespace reco {

  class TransientTrack : public Track {
  public:

    // constructor from persistent track
    TransientTrack( const Track & tk ); 
    TransientTrack( const TrackRef & tk ); 

    TransientTrack( const TransientTrack & tt );
    
    TransientTrack& operator=(const TransientTrack & tt);

    TrajectoryStateOnSurface outermostMeasurementState();

    TrajectoryStateOnSurface innermostMeasurementState();

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const
      {return builder(originalTSCP.theState(), point);}

    TrajectoryStateClosestToPoint impactPointTSCP() const
      {return originalTSCP;}

    TrajectoryStateOnSurface impactPointState() const;

    // access to original persistent track
    const Track & persistentTrack() const { return tk_; }
    const TrackRef & persistentTrackRef() const { return tkr_; }

    TrackCharge charge() const {return charge();}

    bool operator== (const TransientTrack & a) const {return (a.persistentTrackRef()==tkr_);}
    bool operator< (const TransientTrack & a) const 
      {return (originalTSCP.momentum().z()<a.impactPointTSCP().momentum().z());}

  private:

    void calculateStateAtVertex() const;

    const Track & tk_;
    const TrackRef & tkr_;

    TrajectoryStateClosestToPoint originalTSCP;
    mutable bool stateAtVertexAvailable;
    mutable TrajectoryStateOnSurface theStateAtVertex;
    TSCPBuilderNoMaterial builder;


  };

}

#endif
