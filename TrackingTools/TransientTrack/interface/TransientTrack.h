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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

namespace reco {

  class TransientTrack : public Track {
  public:

    // constructor from persistent track
    TransientTrack( const Track & tk , const MagneticField* field); 
    TransientTrack( const TrackRef & tk , const MagneticField* field); 

    TransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrack( const TransientTrack & tt );
    
    TransientTrack& operator=(const TransientTrack & tt);

    void setES(const edm::EventSetup& );

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& );

    TrajectoryStateOnSurface outermostMeasurementState() const;

    TrajectoryStateOnSurface innermostMeasurementState() const;

    TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const
      {return builder(originalTSCP.theState(), point);}

    TrajectoryStateClosestToPoint impactPointTSCP() const
      {return originalTSCP;}

    TrajectoryStateOnSurface impactPointState() const;
    bool impactPointStateAvailable() const {return  stateAtVertexAvailable;}

    // access to original persistent track
    //    const Track & persistentTrack() const { return *tk_; }
    TrackRef persistentTrackRef() const { return tkr_; }

    TrackCharge charge() const {return Track::charge();}

    bool operator== (const TransientTrack & a) const {return (a.persistentTrackRef()==tkr_);}
    bool operator< (const TransientTrack & a) const 
      {return (originalTSCP.momentum().z()<a.impactPointTSCP().momentum().z());}

    const MagneticField* field() const {return theField;}


  private:

    void calculateStateAtVertex() const;

    TrackRef tkr_;
    const MagneticField* theField;

    TrajectoryStateClosestToPoint originalTSCP;
    mutable bool stateAtVertexAvailable;
    mutable TrajectoryStateOnSurface theStateAtVertex;
    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;

  };

}

#endif
