#ifndef TrackingTools_TransientTrack_CandTransientTrack_h
#define TrackingTools_TransientTrack_CandTransientTrack_h

#include <atomic>

  /**
   * Concrete implementation of the TransientTrack for a reco::Track
   */

#include "TrackingTools/TransientTrack/interface/BasicTransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "DataFormats/Candidate/interface/Candidate.h"
namespace reco {

  class CandidatePtrTransientTrack : public Track, public BasicTransientTrack {
  public:

    // constructor from persistent track
    CandidatePtrTransientTrack();
    CandidatePtrTransientTrack( const CandidatePtr & tk , const MagneticField* field);
    CandidatePtrTransientTrack( const CandidatePtr & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    CandidatePtrTransientTrack( const CandidatePtrTransientTrack & tt );

    CandidatePtrTransientTrack& operator=(const CandidatePtrTransientTrack & tt);

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
    TrackRef persistentTrackRef() const { return TrackRef(); }

    TrackBaseRef trackBaseRef() const {return TrackBaseRef();}

    TrackCharge charge() const {return Track::charge();}

    CandidatePtr candidate() const {return ptr_; } 	

    const MagneticField* field() const {return theField;}

    const Track & track() const {return *this;}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const;

  private:

    CandidatePtr ptr_;
    const MagneticField* theField;

    FreeTrajectoryState initialFTS;

    // mutable member data, those should be treated very carefully to guarantee
    // thread safeness of the code by using atomic thread-safe helpers, see below
    mutable TrajectoryStateOnSurface initialTSOS;
    mutable TrajectoryStateClosestToPoint initialTSCP;
    mutable TrajectoryStateClosestToBeamLine trajectoryStateClosestToBeamLine;
    // thread-safe helpers to guarantee proper update of mutable member data
    mutable std::atomic<char> m_TSOS;
    mutable std::atomic<char> m_TSCP;
    mutable std::atomic<char> m_SCTBL;

    TSCPBuilderNoMaterial builder;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    reco::BeamSpot theBeamSpot;

    // to be used to setup thread states of class mutables
    enum CacheStates {kUnset, kSetting, kSet};

  };

}

#endif
