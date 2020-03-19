#ifndef TrackAssociation_ParametersDefinerForTPESProducer_h
#define TrackAssociation_ParametersDefinerForTPESProducer_h

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class ParametersDefinerForTPESProducer : public edm::ESProducer {
  typedef std::unique_ptr<ParametersDefinerForTP> ReturnType;

public:
  ParametersDefinerForTPESProducer(const edm::ParameterSet &p);
  ~ParametersDefinerForTPESProducer() override;
  std::unique_ptr<ParametersDefinerForTP> produce(const TrackAssociatorRecord &);

  edm::ParameterSet pset_;
};

#endif
