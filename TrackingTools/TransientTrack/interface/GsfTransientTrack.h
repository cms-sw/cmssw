#ifndef TrackReco_GsfTransientTrack_h
#define TrackReco_GsfTransientTrack_h

  /**
   * Concrete implementation of the TransientTrack for a multi-state reco::GsfTrack
   */


#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"

namespace reco {

  class GsfTransientTrack : public GsfTrack, public BasicTransientTrack {
  public:

    // constructor from persistent track
    GsfTransientTrack(); 
    GsfTransientTrack( const GsfTrack & tk , const MagneticField* field);
    GsfTransientTrack( const GsfTrack & tk , const double time, const double dtime, const MagneticField* field);
 
    GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field);
    GsfTransientTrack( const GsfTrackRef & tk , const double time, const double dtime, const MagneticField* field);

    GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);
    GsfTransientTrack( const GsfTrackRef & tk , const double time, const double dtime, const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    GsfTransientTrack( const GsfTrack & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);
    GsfTransientTrack( const GsfTrack & tk , const double time, const double dtime, const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);


    GsfTransientTrack( const GsfTransientTrack & tt );
    
    GsfTransientTrack& operator=(const GsfTransientTrack & tt);

    void setES(const edm::EventSetup& );

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& );

    void setBeamSpot(const reco::BeamSpot& beamSpot);

    FreeTrajectoryState initialFreeState() const {return initialFTS;}

    TrajectoryStateOnSurface outermostMeasurementState() const;

    TrajectoryStateOnSurface innermostMeasurementState() const;

   /**
    * The TrajectoryStateClosestToPoint at any point.
    * The inner-most multi-state state will be used for the propagation.
    * The TSCP will be built out of the collapsed mult-component TSOS.
    */
    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const;

   /**
    * The TSOS at any point. The inner-most multi-state state will be used for 
    * the propagation. The TSOS will therefore have multiple components.
    */
    TrajectoryStateOnSurface stateOnSurface(const GlobalPoint & point) const;

   /**
    * The TrajectoryStateClosestToPoint at the initial state, made from the 
    * stored singl-component state.
    */
    TrajectoryStateClosestToPoint impactPointTSCP() const;

   /**
    * The TSOS at the initial state, made from the stored singl-component state.
    */

    TrajectoryStateOnSurface impactPointState() const;

    bool impactPointStateAvailable() const {return  initialTSOSAvailable;}

  /**
   * access to original persistent track
   */
    GsfTrackRef persistentTrackRef() const { return tkr_; }

    TrackBaseRef trackBaseRef() const {return TrackBaseRef(tkr_);}

    TrackCharge charge() const {return GsfTrack::charge();}

    const MagneticField* field() const {return theField;}

    const Track & track() const {return *this;}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const;

    double timeExt() const { return ( hasTime ? timeExt_ : std::numeric_limits<double>::quiet_NaN() ); }

    double dtErrorExt() const { return ( hasTime ? dtErrorExt_ : std::numeric_limits<double>::quiet_NaN() ); }

  private:

    void init();
    void calculateTSOSAtVertex() const;

    GsfTrackRef tkr_;
    bool hasTime;
    double timeExt_, dtErrorExt_;
    const MagneticField* theField;

    FreeTrajectoryState initialFTS;
    mutable bool initialTSOSAvailable, initialTSCPAvailable, blStateAvailable;
    mutable TrajectoryStateOnSurface initialTSOS;
    mutable TrajectoryStateClosestToPoint initialTSCP;
    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    reco::BeamSpot theBeamSpot;
    mutable TrajectoryStateClosestToBeamLine trajectoryStateClosestToBeamLine;

    TransverseImpactPointExtrapolator theTIPExtrapolator;
  

  };

}

#endif
