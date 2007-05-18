#ifndef TrackProducerBase_h
#define TrackProducerBase_h

/** \class TrackProducerBase
 *  Base Class To Produce Tracks
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class TrackerGeometry;

class TrackProducerBase {
public:
  /// Constructor
  TrackProducerBase(bool trajectoryInEvent = false):
    trajectoryInEvent_(trajectoryInEvent){}

  /// Destructor
  virtual ~TrackProducerBase();
  
  /// Get needed services from the Event Setup
  virtual void getFromES(const edm::EventSetup&,
			 edm::ESHandle<TrackerGeometry>& ,
			 edm::ESHandle<MagneticField>& ,
			 edm::ESHandle<TrajectoryFitter>& ,
			 edm::ESHandle<Propagator>& ,
			 edm::ESHandle<TransientTrackingRecHitBuilder>& );

  /// Get TrackCandidateCollection from the Event (needed by TrackProducer)
  virtual void getFromEvt(edm::Event&, edm::Handle<TrackCandidateCollection>&);
  /// Get TrackCollection from the Event (needed by TrackRefitter)
  virtual void getFromEvt(edm::Event&, edm::Handle<reco::TrackCollection>&);

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::TrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);

  /// Method where the procduction take place. To be implemented in concrete classes
  virtual void produce(edm::Event&, const edm::EventSetup&) = 0;

  /// Set parameter set
  void setConf(edm::ParameterSet conf){conf_=conf;}

  /// set label of source collection
  void setSrc(std::string src){src_=src;}

  /// set the producer of source collection
  void setProducer(std::string pro){pro_=pro;}

  /// set the aliases of produced collections
  void setAlias(std::string alias){
    alias.erase(alias.size()-6,alias.size());
    alias_=alias;
  }
 private:
  edm::ParameterSet conf_;
  std::string src_;
  std::string pro_;
  bool trajectoryInEvent_;
  edm::OrphanHandle<reco::TrackCollection> rTracks_;
 protected:
  std::string alias_;
};

#endif
