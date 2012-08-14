#ifndef TrackAssociation_CosmicParametersDefinerForTP_h
#define TrackAssociation_CosmicParametersDefinerForTP_h

/**
 *
 *
 * \author Boris Mangano (UCSD)  5/7/2009
 */

#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>
#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"

class CosmicParametersDefinerForTP : public ParametersDefinerForTP {

 public:
  CosmicParametersDefinerForTP(){};
  virtual ~CosmicParametersDefinerForTP() {};

  virtual ParticleBase::Vector momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticle& tp) const;
  virtual ParticleBase::Point vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticle& tp) const;

};


#endif
