#ifndef TrackReco_TransientTrackFromFTS_h
#define TrackReco_TransientTrackFromFTS_h

  /**
   * Concrete implementation of the TransientTrack for a multi-state reco::GsfTrack
   * To be built through the factory TransientTrackFromFTSFactory or the TransientTrackBuilder
   */

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

namespace reco {

  class TransientTrackFromFTS : public BasicTransientTrack {
  public:

    TransientTrackFromFTS(); 

    TransientTrackFromFTS(const FreeTrajectoryState & fts);
    TransientTrackFromFTS(const FreeTrajectoryState & fts,
	const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrackFromFTS( const TransientTrackFromFTS & tt );
    
    TransientTrackFromFTS& operator=(const TransientTrackFromFTS & tt);

    void setES(const edm::EventSetup& );

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& );

    void setBeamSpot(const reco::BeamSpot& beamSpot);

    FreeTrajectoryState initialFreeState() const {return initialFTS;}

    TrajectoryStateOnSurface outermostMeasurementState() const;

    TrajectoryStateOnSurface innermostMeasurementState() const;

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const
      {return builder(initialFTS, point);}

   /**
    * The TSOS at any point. The initial state will be used for the propagation.
    */
    TrajectoryStateOnSurface stateOnSurface(const GlobalPoint & point) const;

    TrajectoryStateClosestToPoint impactPointTSCP() const;

    TrajectoryStateOnSurface impactPointState() const;

    bool impactPointStateAvailable() const {return  initialTSOSAvailable;}

    TrackCharge charge() const {return initialFTS.charge();}

    const MagneticField* field() const {return theField;}

    const Track & track() const;

    TrackBaseRef trackBaseRef() const { return TrackBaseRef();}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const;

  private:

    void calculateTSOSAtVertex() const;

    FreeTrajectoryState initialFTS;
    const MagneticField* theField;
    mutable bool initialTSOSAvailable, initialTSCPAvailable, trackAvailable, blStateAvailable;
    mutable TrajectoryStateOnSurface initialTSOS;
    mutable TrajectoryStateClosestToPoint initialTSCP;
    mutable Track theTrack;
    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    reco::BeamSpot theBeamSpot;
    mutable TrajectoryStateClosestToBeamLine trajectoryStateClosestToBeamLine;

  };

}

#endif
