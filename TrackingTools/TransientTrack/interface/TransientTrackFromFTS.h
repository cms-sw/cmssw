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
    TransientTrackFromFTS(const FreeTrajectoryState & fts, const double time, const double dtime);
    
    TransientTrackFromFTS(const FreeTrajectoryState & fts,
	const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);
    TransientTrackFromFTS(const FreeTrajectoryState & fts,
                          const double time,
                          const double dtime, 
                          const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrackFromFTS( const TransientTrackFromFTS & tt );
    
    TransientTrackFromFTS& operator=(const TransientTrackFromFTS & tt);

    void setES(const edm::EventSetup& ) override;

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& ) override;

    void setBeamSpot(const reco::BeamSpot& beamSpot) override;

    FreeTrajectoryState initialFreeState() const override {return initialFTS;}

    TrajectoryStateOnSurface outermostMeasurementState() const override;

    TrajectoryStateOnSurface innermostMeasurementState() const override;

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const override
      {return builder(initialFTS, point);}

   /**
    * The TSOS at any point. The initial state will be used for the propagation.
    */
    TrajectoryStateOnSurface stateOnSurface(const GlobalPoint & point) const override;

    TrajectoryStateClosestToPoint impactPointTSCP() const override;

    TrajectoryStateOnSurface impactPointState() const override;

    bool impactPointStateAvailable() const override {return  initialTSOSAvailable;}

    TrackCharge charge() const override {return initialFTS.charge();}

    const MagneticField* field() const override {return theField;}

    const Track & track() const override;

    TrackBaseRef trackBaseRef() const override { return TrackBaseRef();}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const override;
    
    double timeExt() const override { return ( hasTime ? timeExt_ : std::numeric_limits<double>::quiet_NaN() ); }
    double dtErrorExt() const override { return ( hasTime ? dtErrorExt_ : std::numeric_limits<double>::quiet_NaN() ); }

  private:

    void calculateTSOSAtVertex() const;

    FreeTrajectoryState initialFTS;
    bool hasTime;
    double timeExt_;
    double dtErrorExt_;
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
