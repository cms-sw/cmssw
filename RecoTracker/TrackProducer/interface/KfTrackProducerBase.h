#ifndef KfTrackProducerBase_h
#define KfTrackProducerBase_h

/** \class KfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class Trajectory;

class KfTrackProducerBase : public TrackProducerBase<reco::Track> {
public:
  /// Constructor
  explicit KfTrackProducerBase(bool trajectoryInEvent, bool split)
      : TrackProducerBase<reco::Track>(trajectoryInEvent), rekeyClusterRefs_(false), useSplitting(split) {}

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
                        const Propagator* prop,
                        const MeasurementTracker* measTk,
                        std::unique_ptr<TrackingRecHitCollection>&,
                        std::unique_ptr<reco::TrackCollection>&,
                        std::unique_ptr<reco::TrackExtraCollection>&,
                        std::unique_ptr<std::vector<Trajectory> >&,
                        std::unique_ptr<std::vector<int> >&,
                        AlgoProductCollection&,
                        TransientTrackingRecHitBuilder const*,
                        const TrackerTopology* ttopo,
                        //allow to fill different tracks collections if necessary ::
                        //0: not needed
                        //1: Before DAF
                        //2: After DAF
                        int BeforeOrAfter = 0);

  //  void setSecondHitPattern(Trajectory* traj, reco::Track& track);
protected:
  /// Sets the information on cluster removal, and turns it on
  void setClusterRemovalInfo(const edm::InputTag& clusterRemovalInfo) {
    rekeyClusterRefs_ = true;
    clusterRemovalInfo_ = clusterRemovalInfo;
  }

  bool rekeyClusterRefs_;
  edm::InputTag clusterRemovalInfo_;

private:
  bool useSplitting;
};

#endif
