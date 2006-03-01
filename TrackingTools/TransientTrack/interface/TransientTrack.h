#ifndef TrackReco_TransientTrack_h
#define TrackReco_TransientTrack_h
//
// Definition of Transient Track class for 
// reconstruction posterior to track reconstruction (vertexing, b-tagging...)
//

#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {

  class TransientTrack : public Track {
  public:

    // constructor from persistent track
    TransientTrack( const Track & tk ); 

    TrajectoryStateOnSurface outermostMeasurementState();

    TrajectoryStateOnSurface innermostMeasurementState();

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & xyz );

    TrajectoryStateOnSurface impactPointState();

    // access to original persistent track
    const Track & persistentTrack() { return tk_; }

  private:

    const Track & tk_;

  };

}

#endif
