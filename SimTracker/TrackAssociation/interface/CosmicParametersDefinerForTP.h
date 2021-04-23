#ifndef TrackAssociation_CosmicParametersDefinerForTP_h
#define TrackAssociation_CosmicParametersDefinerForTP_h

/**
 *
 *
 * \author Boris Mangano (UCSD)  5/7/2009
 */

#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>

#include <memory>

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

  std::tuple<TrackingParticle::Vector, TrackingParticle::Point> momentumAndVertex(
      const edm::Event &iEvent, const edm::EventSetup &iSetup, const TrackingParticleRef &tpr) const override {
    return std::make_tuple(momentum(iEvent, iSetup, tpr), vertex(iEvent, iSetup, tpr));
  }

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
    return std::make_unique<CosmicParametersDefinerForTP>(*this);
  }

private:
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
};

#endif
