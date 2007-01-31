#ifndef TrackReco_GsfTransientTrack_h
#define TrackReco_GsfTransientTrack_h
//
// Definition of Transient Track class for 
// reconstruction posterior to track reconstruction (vertexing, b-tagging...)
//

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// #include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
// #include "FWCore/Framework/interface/EventSetup.h"
// #include "FWCore/Framework/interface/ESHandle.h"
// #include "MagneticField/Engine/interface/MagneticField.h"
// #include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

namespace reco {

  class GsfTransientTrack : public GsfTrack, public BasicTransientTrack {
  public:

    // constructor from persistent track
    GsfTransientTrack(); 
    GsfTransientTrack( const GsfTrack & tk , const MagneticField* field); 
    GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field); 

    GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    GsfTransientTrack( const GsfTrack & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    GsfTransientTrack( const GsfTransientTrack & tt );
    
    GsfTransientTrack& operator=(const GsfTransientTrack & tt);

    void setES(const edm::EventSetup& );

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& );

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

    // access to original persistent track
    //    const GsfTrack & persistentTrack() const { return *tk_; }
    GsfTrackRef persistentTrackRef() const { return tkr_; }

    TrackCharge charge() const {return GsfTrack::charge();}

//     bool operator== (const TransientTrack & a) const {return (a.persistentTrackRef()==tkr_);}
//     bool operator< (const TransientTrack & a) const 
//       {return (initialFTS.momentum().z()<a.initialFreeState().momentum().z());}

    const MagneticField* field() const {return theField;}


  private:

    void init();
    void calculateTSOSAtVertex() const;

    GsfTrackRef tkr_;
    const MagneticField* theField;

    FreeTrajectoryState initialFTS;
    mutable bool initialTSOSAvailable, initialTSCPAvailable;
    mutable TrajectoryStateOnSurface initialTSOS;
    mutable TrajectoryStateClosestToPoint initialTSCP;
    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    TransverseImpactPointExtrapolator* theTIPExtrapolator;
    Propagator* thePropagator;

  };

}

#endif
