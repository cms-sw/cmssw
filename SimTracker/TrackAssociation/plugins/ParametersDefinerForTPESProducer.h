#ifndef TrackAssociation_ParametersDefinerForTPESProducer_h
#define TrackAssociation_ParametersDefinerForTPESProducer_h


#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <memory>

class  ParametersDefinerForTPESProducer: public edm::ESProducer{
  typedef std::shared_ptr<ParametersDefinerForTP> ReturnType;

 public:
  ParametersDefinerForTPESProducer(const edm::ParameterSet & p);
  virtual ~ParametersDefinerForTPESProducer(); 
  std::shared_ptr<ParametersDefinerForTP> produce(const TrackAssociatorRecord &);

  edm::ParameterSet pset_;

};


#endif
