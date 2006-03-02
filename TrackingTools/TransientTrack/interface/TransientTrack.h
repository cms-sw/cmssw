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

    TrajectoryStateOnSurface outermostMeasurementState();

    TrajectoryStateOnSurface innermostMeasurementState();

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point )
      {return builder(originalTSCP.theState(), point);}

    TrajectoryStateClosestToPoint impactPointTSCP()
      {return originalTSCP;}

    TrajectoryStateOnSurface impactPointState();

    // access to original persistent track
    const Track & persistentTrack() { return tk_; }

  private:

    void calculateStateAtVertex();

    const Track & tk_;

    TrajectoryStateClosestToPoint originalTSCP;
    bool stateAtVertexAvailable;
    TrajectoryStateOnSurface theStateAtVertex;
    TSCPBuilderNoMaterial builder;


  };

}

#endif
