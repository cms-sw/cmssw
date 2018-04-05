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
    
    CandidatePtrTransientTrack( const CandidatePtr & tk, const MagneticField* field);
    CandidatePtrTransientTrack( const CandidatePtr & tk, const double time, const double dtime, const MagneticField* field);

    CandidatePtrTransientTrack( const CandidatePtr & tk, const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);
    CandidatePtrTransientTrack( const CandidatePtr & tk, const double time, const double dtime, const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);

    CandidatePtrTransientTrack( const CandidatePtrTransientTrack & tt );

    CandidatePtrTransientTrack& operator=(const CandidatePtrTransientTrack & tt);

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

    bool impactPointStateAvailable() const override {return (m_TSOS.load()==kSet ? true : false); }

  /**
   * access to original persistent track
   */
    TrackRef persistentTrackRef() const { return TrackRef(); }

    TrackBaseRef trackBaseRef() const override {return TrackBaseRef();}

    TrackCharge charge() const override {return Track::charge();}

    CandidatePtr candidate() const override {return ptr_; } 	

    const MagneticField* field() const override {return theField;}

    const Track & track() const override {return *this;}

    TrajectoryStateClosestToBeamLine stateAtBeamLine() const override;
    
    double timeExt() const override { return ( hasTime ? timeExt_ : std::numeric_limits<double>::quiet_NaN() ); }
    double dtErrorExt() const override { return ( hasTime ? dtErrorExt_ : std::numeric_limits<double>::quiet_NaN() ); }

  private:

    CandidatePtr ptr_;
    bool hasTime;
    double timeExt_, dtErrorExt_;
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
