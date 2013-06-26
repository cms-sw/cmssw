#ifndef TrackReco_BasicTransientTrack_h
#define TrackReco_BasicTransientTrack_h

/**
 * Abstract Base class for reference counted TransientTrack
 */

#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToBeamLine.h"

namespace reco {

  class BasicTransientTrack : public ReferenceCountedInEvent {
  public:

    typedef BasicTransientTrack                              BTT;
    typedef ProxyBase< BTT, CopyUsingClone<BTT> >        Proxy;
    typedef ReferenceCountingPointer<BasicTransientTrack>    RCPtr;

  private:
    friend class ProxyBase< BTT, CopyUsingClone<BTT> >;
    friend class ReferenceCountingPointer<BasicTransientTrack>;

  public:

    virtual ~BasicTransientTrack() {}

    virtual void setES(const edm::EventSetup& es) = 0;

    virtual void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg) = 0;

    virtual void setBeamSpot(const reco::BeamSpot& beamSpot) = 0;

    virtual FreeTrajectoryState initialFreeState() const = 0;

    virtual TrajectoryStateOnSurface outermostMeasurementState() const = 0;

    virtual TrajectoryStateOnSurface innermostMeasurementState() const = 0;

    virtual TrajectoryStateClosestToPoint 
      trajectoryStateClosestToPoint( const GlobalPoint & point ) const = 0;

    virtual TrajectoryStateOnSurface stateOnSurface(const GlobalPoint & point) const = 0;

    virtual TrajectoryStateClosestToPoint impactPointTSCP() const = 0;

    virtual TrajectoryStateOnSurface impactPointState() const = 0;
    virtual bool impactPointStateAvailable() const = 0;

    virtual TrackBaseRef trackBaseRef() const = 0;

    virtual TrackCharge charge() const = 0;

//     virtual bool operator== (const TransientTrack & a) const = 0;
//     virtual bool operator< (const TransientTrack & a) const = 0;

    virtual const MagneticField* field() const = 0;

    virtual const Track & track() const = 0;

    virtual TrajectoryStateClosestToBeamLine stateAtBeamLine() const = 0;

  };

}

#endif
