#ifndef TrackAssociation_CosmicParametersDefinerForTPESProducer_h
#define TrackAssociation_CosmicParametersDefinerForTPESProducer_h


#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <memory>

class  CosmicParametersDefinerForTPESProducer: public edm::ESProducer{
  typedef std::shared_ptr<CosmicParametersDefinerForTP> ReturnType;

 public:
  CosmicParametersDefinerForTPESProducer(const edm::ParameterSet & p);
  virtual ~CosmicParametersDefinerForTPESProducer(); 
  std::shared_ptr<CosmicParametersDefinerForTP> produce(const TrackAssociatorRecord &);

};


#endif
