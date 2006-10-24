#ifndef GsfTrackProducerBase_h
#define GsfTrackProducerBase_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/TrackProducer/interface/GsfTrackProducerAlgorithm.h"
#include "DataFormats/TrackReco/interface/GsfComponent5D.h"

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;
class TrackerGeometry;

class GsfTrackProducerBase {
public:
  GsfTrackProducerBase(bool trajectoryInEvent = false):
    trajectoryInEvent_(trajectoryInEvent){}

  virtual ~GsfTrackProducerBase();
  
  virtual void getFromES(const edm::EventSetup&,
			 edm::ESHandle<TrackerGeometry>& ,
			 edm::ESHandle<MagneticField>& ,
			 edm::ESHandle<TrajectoryFitter>& ,
			 edm::ESHandle<Propagator>& ,
			 edm::ESHandle<TransientTrackingRecHitBuilder>& );

  virtual void getFromEvt(edm::Event&, edm::Handle<TrackCandidateCollection>&);
  virtual void getFromEvt(edm::Event&, edm::Handle<reco::TrackCollection>&);

  typedef GsfTrackProducerAlgorithm::AlgoProductCollection AlgoProductCollection;
  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::GsfTrackCollection>&,
			std::auto_ptr<reco::GsfTrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);

  virtual void produce(edm::Event&, const edm::EventSetup&) = 0;

  void setConf(edm::ParameterSet conf){conf_=conf;}
  //edm::ParameterSet conf(){return conf;}
  void setSrc(std::string src){src_=src;}
  void setAlias(std::string alias){
    alias.erase(alias.size()-6,alias.size());
    alias_=alias;
  }
private:
  void fillStates (TrajectoryStateOnSurface tsos, std::vector<reco::GsfComponent5D>& states) const;
 private:
  edm::ParameterSet conf_;
  std::string src_;
  bool trajectoryInEvent_;
 protected:
  std::string alias_;

};

#endif
