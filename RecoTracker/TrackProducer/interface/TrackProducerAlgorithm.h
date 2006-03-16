#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducerAlgorithm
// 
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id$
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

class MagneticField;
class TrackingGeometry;
class KFTrajectoryFitter;

class TrackProducerAlgorithm {
  
 public:
  TrackProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)
    { }

  ~TrackProducerAlgorithm() {}
  
  void run(const TrackingGeometry *, 
	    const MagneticField *, 
	    TrackCandidateCollection&,
	    const KFTrajectoryFitter *,
	    std::auto_ptr<reco::TrackCollection>&, 
	    std::auto_ptr<reco::TrackExtraCollection>&);
  
 private:
  edm::ParameterSet conf_;
};

#endif
