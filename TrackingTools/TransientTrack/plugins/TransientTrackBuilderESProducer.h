#ifndef TrackingTools_ESProducers_TransientTrackBuilderESProducer_h
#define TrackingTools_ESProducers_TransientTrackBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <boost/shared_ptr.hpp>

class  TransientTrackBuilderESProducer: public edm::ESProducer{
 public:
  TransientTrackBuilderESProducer(const edm::ParameterSet & p);
  virtual ~TransientTrackBuilderESProducer(); 
  boost::shared_ptr<TransientTrackBuilder> produce(const TransientTrackRecord &);
 private:
  boost::shared_ptr<TransientTrackBuilder> _builder;
  edm::ParameterSet pset_;
};


#endif




