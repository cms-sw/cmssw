#ifndef TrackAssociation_ParametersDefinerForTP_h
#define TrackAssociation_ParametersDefinerForTP_h

/**
 *
 *
 * \author Boris Mangano (UCSD)  5/7/2009
 */

#include <SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"      
#include "FWCore/Framework/interface/EventSetup.h"

class ParametersDefinerForTP {

 public:
  ParametersDefinerForTP(){};
  virtual ~ParametersDefinerForTP() {};

  virtual ParticleBase::Vector momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const ParticleBase& tp) const;
  virtual ParticleBase::Point vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const ParticleBase& tp) const;

  virtual ParticleBase::Vector momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Candidate& tp) const {
    return momentum(iEvent, iSetup, ParticleBase(tp.charge(),tp.p4(),tp.vertex()));
  }
  virtual ParticleBase::Point vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Candidate& tp) const {
    return vertex(iEvent, iSetup, ParticleBase(tp.charge(),tp.p4(),tp.vertex()));
  }

};


#endif
