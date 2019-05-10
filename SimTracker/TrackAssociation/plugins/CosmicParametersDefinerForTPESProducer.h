#ifndef TrackAssociation_CosmicParametersDefinerForTPESProducer_h
#define TrackAssociation_CosmicParametersDefinerForTPESProducer_h

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class CosmicParametersDefinerForTPESProducer : public edm::ESProducer {
  typedef std::unique_ptr<CosmicParametersDefinerForTP> ReturnType;

public:
  CosmicParametersDefinerForTPESProducer(const edm::ParameterSet &p);
  ~CosmicParametersDefinerForTPESProducer() override;
  std::unique_ptr<CosmicParametersDefinerForTP> produce(const TrackAssociatorRecord &);
};

#endif
