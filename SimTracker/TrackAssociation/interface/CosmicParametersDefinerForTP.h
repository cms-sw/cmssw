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

  virtual TrackingParticle::Vector momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticleRef tpr) const;
  virtual TrackingParticle::Point vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticleRef tpr) const;

  virtual TrackingParticle::Vector momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
	const Charge ch, const Point & vertex, const LorentzVector& lv) const {
    return TrackingParticle::Vector();
  }

  virtual TrackingParticle::Point vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup,
	const Charge ch, const Point & vertex, const LorentzVector& lv) const {
    return TrackingParticle::Point();
  }

  void initEvent(edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssocToSet) const {
    simHitsTPAssoc = simHitsTPAssocToSet;
  }

 private:
  mutable edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
};


#endif
