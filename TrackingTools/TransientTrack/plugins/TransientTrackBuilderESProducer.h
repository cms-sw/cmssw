#ifndef TrackingTools_ESProducers_TransientTrackBuilderESProducer_h
#define TrackingTools_ESProducers_TransientTrackBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <memory>

class  TransientTrackBuilderESProducer: public edm::ESProducer{
 public:
  TransientTrackBuilderESProducer(const edm::ParameterSet & p);
  ~TransientTrackBuilderESProducer() override; 
  std::shared_ptr<TransientTrackBuilder> produce(const TransientTrackRecord &);
 private:
  std::shared_ptr<TransientTrackBuilder> _builder;
  edm::ParameterSet pset_;
};


#endif




