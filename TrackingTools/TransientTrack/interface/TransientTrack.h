#ifndef TrackReco_TransientTrack_h
#define TrackReco_TransientTrack_h
//
// Definition of Transient Track class for 
// reconstruction posterior to track reconstruction (vertexing, b-tagging...)
//

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"

#include "DataFormats/TrackReco/interface/Track.h"
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// #include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
// #include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
// #include "FWCore/Framework/interface/EventSetup.h"
// #include "FWCore/Framework/interface/ESHandle.h"
// #include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
// #include "MagneticField/Engine/interface/MagneticField.h"
// #include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

namespace reco {

  class TransientTrack : private  BasicTransientTrack::Proxy {

    typedef BasicTransientTrack::Proxy             Base;

  public:

    // constructor from persistent track
    TransientTrack(); 
    TransientTrack( const Track & tk , const MagneticField* field); 
    TransientTrack( const TrackRef & tk , const MagneticField* field); 

    TransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrack( BasicTransientTrack * btt ) : Base(btt) {}

//     TransientTrack& operator=(const TransientTrack & tt);

    void setES(const edm::EventSetup& es) {sharedData().setES(es);}

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg)
    	{sharedData().setTrackingGeometry(tg);}

    FreeTrajectoryState initialFreeState() const {return data().initialFreeState();}

    TrajectoryStateOnSurface outermostMeasurementState() const
	{return data().outermostMeasurementState();}

    TrajectoryStateOnSurface innermostMeasurementState() const
	{return data().innermostMeasurementState();}

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const
	{return data().trajectoryStateClosestToPoint(point);}

    TrajectoryStateOnSurface stateOnSurface(const GlobalPoint & point) const
	{return data().stateOnSurface(point);}

    TrajectoryStateClosestToPoint impactPointTSCP() const
	{return data().impactPointTSCP();}

    TrajectoryStateOnSurface impactPointState() const
	{return data().impactPointState();}

    bool impactPointStateAvailable() const
	{return data().impactPointStateAvailable();}

    // access to original persistent track
    //    const Track & persistentTrack() const { return *tk_; }
//     TrackRef persistentTrackRef() const { return tkr_; }

    TrackCharge charge() const {return data().charge();}

    bool operator== (const TransientTrack & other) const
	{return &(data()) == &(other.data());}
    // {return (a.persistentTrackRef()==tkr_);}

    bool operator< (const TransientTrack & other) const 
	{return &(data()) < &(other.data());}
    // {return (initialFTS.momentum().z()<a.initialFreeState().momentum().z());}

    const MagneticField* field() const {return data().field();}

    const BasicTransientTrack* basicTransientTrack() const {return &(data());}

  };

}

#endif
