#ifndef TrackReco_TrackTransientTrack_h
#define TrackReco_TrackTransientTrack_h

#include <atomic>

  /**
   * Concrete implementation of the TransientTrack for a reco::Track
   */

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

namespace reco {

  class TrackTransientTrack : public Track, public BasicTransientTrack {
  public:

    // constructor from persistent track
    TrackTransientTrack();
    TrackTransientTrack( const Track & tk , const MagneticField* field);
    TrackTransientTrack( const TrackRef & tk , const MagneticField* field);

    TrackTransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TrackTransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TrackTransientTrack( const TrackTransientTrack & tt );

    TrackTransientTrack& operator=(const TrackTransientTrack & tt);

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

    bool impactPointStateAvailable() const {return (m_TSOS.load()==kSet ? true : false); }

  /**
   * access to original persistent track
   */
    TrackRef persistentTrackRef() const { return tkr_; }

    TrackBaseRef trackBaseRef() const {return TrackBaseRef(tkr_);}

    TrackCharge charge() const {return Track::charge();}

    const MagneticField* field() const {return theField;}

    const Track & track() const {return *this;}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const;

  private:

    TrackRef tkr_;
    const MagneticField* theField;

    FreeTrajectoryState initialFTS;
    mutable TrajectoryStateOnSurface initialTSOS;
    mutable std::atomic<char> m_TSOS;
    mutable TrajectoryStateClosestToPoint initialTSCP;
    mutable std::atomic<char> m_TSCP;
    mutable TrajectoryStateClosestToBeamLine trajectoryStateClosestToBeamLine;
    mutable std::atomic<char> m_SCTBL;

    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    reco::BeamSpot theBeamSpot;

    enum KlassStates {kUnset, kSetting, kSet};

  };

}

#endif
