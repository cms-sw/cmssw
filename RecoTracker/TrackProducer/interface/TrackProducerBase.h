#ifndef TrackProducerBase_h
#define TrackProducerBase_h

/** \class TrackProducerBase
 *  Base Class To Produce Tracks
 *
 *  \author cerati
 */


#include "AlgoProductTraits.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/ClusterRemovalInfo.h"
#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>

#include<tuple>

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class TrackerGeometry;
class TrajectoryFitter;
class TransientTrackingRecHitBuilder;
class NavigationSchool;

template <class T>
class TrackProducerBase : public AlgoProductTraits<T> {
public:
  using Base = AlgoProductTraits<T>;
  using TrackView = typename Base::TrackView;
  using TrackCollection = typename Base::TrackCollection;
  using AlgoProductCollection = typename Base::AlgoProductCollection;
public:

  /// Constructor
  TrackProducerBase(bool trajectoryInEvent = false):
     trajectoryInEvent_(trajectoryInEvent),
        rekeyClusterRefs_(false) {}

  /// Destructor
  virtual ~TrackProducerBase() noexcept(false);
  
  /// Get needed services from the Event Setup
  virtual void getFromES(const edm::EventSetup&,
			 edm::ESHandle<TrackerGeometry>& ,
			 edm::ESHandle<MagneticField>& ,
			 edm::ESHandle<TrajectoryFitter>& ,
			 edm::ESHandle<Propagator>& ,
			 edm::ESHandle<MeasurementTracker>& ,
			 edm::ESHandle<TransientTrackingRecHitBuilder>& );

  /// Get TrackCandidateCollection from the Event (needed by TrackProducer)
  virtual void getFromEvt(edm::Event&, edm::Handle<TrackCandidateCollection>&, reco::BeamSpot&);
  /// Get TrackCollection from the Event (needed by TrackRefitter)
  virtual void getFromEvt(edm::Event&, edm::Handle<TrackView>&, reco::BeamSpot&);

  /// Method where the procduction take place. To be implemented in concrete classes
  virtual void produce(edm::Event&, const edm::EventSetup&) = 0;

  /// Set parameter set
  void setConf(const edm::ParameterSet& conf){conf_=conf;}

  /// set label of source collection
  void setSrc(const edm::EDGetToken& src, const edm::EDGetTokenT<reco::BeamSpot>& bsSrc, const edm::EDGetTokenT<MeasurementTrackerEvent> &mteSrc) {
    src_ = src; bsSrc_ = bsSrc; mteSrc_ = mteSrc;
  }

  /// set the aliases of produced collections
  void setAlias(std::string alias){
    alias.erase(alias.size()-6,alias.size());
    alias_=alias;
  }

  /// Sets the information on cluster removal, and turns it on
  void setClusterRemovalInfo(const edm::InputTag &clusterRemovalInfo) {
    rekeyClusterRefs_ = true;
    clusterRemovalInfo_ = clusterRemovalInfo;
  }

  void setSecondHitPattern(Trajectory* traj, T& track, 
			   const Propagator* prop, const MeasurementTrackerEvent* measTk,
                           const TrackerTopology* ttopo);

  const edm::ParameterSet& getConf() const {return conf_;}
 protected:
  edm::ParameterSet conf_;
  edm::EDGetToken src_;
 protected:
  std::string alias_;
  bool trajectoryInEvent_;
  edm::OrphanHandle<TrackCollection> rTracks_;
  edm::EDGetTokenT<reco::BeamSpot> bsSrc_;
  edm::EDGetTokenT<MeasurementTrackerEvent> mteSrc_;

  bool rekeyClusterRefs_;
  edm::InputTag clusterRemovalInfo_;

  edm::ESHandle<NavigationSchool> theSchool;

};

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.icc"

#endif
