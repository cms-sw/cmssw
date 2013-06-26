#ifndef TrackReco_TransientTrack_h
#define TrackReco_TransientTrack_h


  /**
   * Definition of Transient Track class to be used for higher-level reconstruction
   *  (vertexing, b-tagging...). It allows access to several services that the 
   *  DataFormat tracks can not access (magnetic field, geometry)
   */


#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/Common/interface/RefToBase.h" 

namespace reco {

  class TransientTrack : private  BasicTransientTrack::Proxy {

    typedef BasicTransientTrack::Proxy             Base;

  public:

    TransientTrack() noexcept {}

    explicit TransientTrack( BasicTransientTrack * btt ) noexcept : Base(btt) {}

    ~TransientTrack() noexcept {}


#if defined( __GXX_EXPERIMENTAL_CXX0X__)

    TransientTrack(TransientTrack const & rh) noexcept :
      Base(rh){}
    
    
    TransientTrack(TransientTrack && rh) noexcept :
      Base(std::move(rh)){}
    
    TransientTrack & operator=(TransientTrack && rh) noexcept {
      Base::operator=(std::move(rh));
      return *this;
    }
    
    TransientTrack & operator=(TransientTrack const & rh) noexcept {
      Base::operator=(rh);
      return *this;
    }

#endif
  
    void swap(TransientTrack & rh) noexcept {
      Base::swap(rh);
    }

    TransientTrack( const Track & tk , const MagneticField* field); 
    TransientTrack( const TrackRef & tk , const MagneticField* field); 

    TransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    TransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);



    void setES(const edm::EventSetup& es) {sharedData().setES(es);}

    void setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg)
    	{sharedData().setTrackingGeometry(tg);}

    void setBeamSpot(const reco::BeamSpot& beamSpot) 
    	{sharedData().setBeamSpot(beamSpot);}

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

    TrackCharge charge() const {return data().charge();}

    bool operator== (const TransientTrack & other) const
	{return &(data()) == &(other.data());}
    // {return (a.persistentTrackRef()==tkr_);}

    bool operator< (const TransientTrack & other) const 
	{return &(data()) < &(other.data());}
    // {return (initialFTS.momentum().z()<a.initialFreeState().momentum().z());}

    const MagneticField* field() const {return data().field();}

    const BasicTransientTrack* basicTransientTrack() const {return &(data());}

    const Track & track() const {return data().track();}

    TrackBaseRef trackBaseRef() const {return data().trackBaseRef();}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const
	{return data().stateAtBeamLine();}

// Methods forwarded to original track.

    /// first iterator to RecHits
    trackingRecHit_iterator recHitsBegin() const { return track().recHitsBegin(); }
    /// last iterator to RecHits
    trackingRecHit_iterator recHitsEnd() const { return track().recHitsEnd(); }
    /// get n-th recHit
    TrackingRecHitRef recHit( size_t i ) const { return track().recHit( i ); }
    /// number of RecHits
    size_t recHitsSize() const { return track().recHitsSize(); }
    //  hit pattern
    const HitPattern & hitPattern() const { return track().hitPattern(); }
    /// number of hits found 
    unsigned short numberOfValidHits() const { return track().hitPattern().numberOfValidHits(); }
    /// number of hits lost
    unsigned short numberOfLostHits() const { return track().hitPattern().numberOfLostHits(); }
    /// chi-squared of the fit
    double chi2() const { return track().chi2(); }
    /// number of degrees of freedom of the fit
    double ndof() const { return track().ndof(); }
    /// chi-squared divided by n.d.o.f.
    double normalizedChi2() const { return track().chi2() / track().ndof(); }

    /// Make the ReferenceCountingProxy method to check validity public
    bool isValid() const {return Base::isValid() ;}

  };

}

#endif
