#ifndef TrackAssociation_CosmicParametersDefinerForTPESProducer_h
#define TrackAssociation_CosmicParametersDefinerForTPESProducer_h


#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <boost/shared_ptr.hpp>

class  CosmicParametersDefinerForTPESProducer: public edm::ESProducer{
  typedef boost::shared_ptr<CosmicParametersDefinerForTP> ReturnType;

 public:
  CosmicParametersDefinerForTPESProducer(const edm::ParameterSet & p);
  virtual ~CosmicParametersDefinerForTPESProducer(); 
  boost::shared_ptr<CosmicParametersDefinerForTP> produce(const TrackAssociatorRecord &);

};


#endif
