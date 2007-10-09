#ifndef GsfTrackProducer_h
#define GsfTrackProducer_h

#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"

class GsfTrackProducer : public GsfTrackProducerBase, public edm::EDProducer {
public:

  explicit GsfTrackProducer(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

//   virtual void putInEvt(edm::Event&,
// 			std::auto_ptr<TrackingRecHitCollection>&,
// 			std::auto_ptr<reco::GsfTrackCollection>&,
// 			std::auto_ptr<reco::TrackExtraCollection>&,
// 			std::auto_ptr<reco::GsfTrackExtraCollection>&,
// 			std::auto_ptr<std::vector<Trajectory> >&,
// 			AlgoProductCollection&);

//   std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);
// protected:
//   void fillStates (TrajectoryStateOnSurface tsos, std::vector<reco::GsfComponent5D>& states) const;

private:
  TrackProducerAlgorithm<reco::GsfTrack> theAlgo;

};

#endif
