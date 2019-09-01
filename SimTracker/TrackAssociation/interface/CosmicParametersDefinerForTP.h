#ifndef TrackAssociation_CosmicParametersDefinerForTP_h
#define TrackAssociation_CosmicParametersDefinerForTP_h

/**
 *
 *
 * \author Boris Mangano (UCSD)  5/7/2009
 */

#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>

class CosmicParametersDefinerForTP : public ParametersDefinerForTP {
public:
  CosmicParametersDefinerForTP(){};
  ~CosmicParametersDefinerForTP() override{};

  TrackingParticle::Vector momentum(const edm::Event &iEvent,
                                    const edm::EventSetup &iSetup,
                                    const TrackingParticleRef &tpr) const override;
  TrackingParticle::Point vertex(const edm::Event &iEvent,
                                 const edm::EventSetup &iSetup,
                                 const TrackingParticleRef &tpr) const override;

  TrackingParticle::Vector momentum(const edm::Event &iEvent,
                                    const edm::EventSetup &iSetup,
                                    const Charge ch,
                                    const Point &vertex,
                                    const LorentzVector &lv) const override {
    return TrackingParticle::Vector();
  }

  TrackingParticle::Point vertex(const edm::Event &iEvent,
                                 const edm::EventSetup &iSetup,
                                 const Charge ch,
                                 const Point &vertex,
                                 const LorentzVector &lv) const override {
    return TrackingParticle::Point();
  }

  void initEvent(edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssocToSet) override {
    simHitsTPAssoc = simHitsTPAssocToSet;
  }

  std::unique_ptr<ParametersDefinerForTP> clone() const override {
    return std::unique_ptr<CosmicParametersDefinerForTP>(new CosmicParametersDefinerForTP(*this));
  }

private:
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
};

#endif
