#ifndef TrackProducerBase_h
#define TrackProducerBase_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducerBase
// 
//
// Description: Base Class To Produce Tracks
//
//
// Original Author:  Giuseppe Cerati
//         Created:  Wed May  10 12:29:31 CET 2006
//

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
  TrackProducerBase(bool trajectoryInEvent = false):
    trajectoryInEvent_(trajectoryInEvent){}

  virtual ~TrackProducerBase();
  
  virtual void getFromES(const edm::EventSetup&,
			 edm::ESHandle<TrackerGeometry>& ,
			 edm::ESHandle<MagneticField>& ,
			 edm::ESHandle<TrajectoryFitter>& ,
			 edm::ESHandle<Propagator>& ,
			 edm::ESHandle<TransientTrackingRecHitBuilder>& );

  virtual void getFromEvt(edm::Event&, edm::Handle<TrackCandidateCollection>&);
  virtual void getFromEvt(edm::Event&, edm::Handle<reco::TrackCollection>&);

  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::TrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);

  virtual void produce(edm::Event&, const edm::EventSetup&) = 0;

  void setConf(edm::ParameterSet conf){conf_=conf;}
  //edm::ParameterSet conf(){return conf;}
  void setSrc(std::string src){src_=src;}
  void setProducer(std::string pro){pro_=pro;}
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
