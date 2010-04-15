#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorTK.h"

BetaCalculatorTK::BetaCalculatorTK(const edm::ParameterSet& iConfig){
  m_dedxEstimatorTag     = iConfig.getParameter<edm::InputTag>("dedxEstimator");
  m_dedxDiscriminatorTag = iConfig.getParameter<edm::InputTag>("dedxDiscriminator");
}


void BetaCalculatorTK::addInfoToCandidate(HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if(!candidate.hasTrackRef())return;

   edm::Handle<DeDxDataValueMap> EstimatorH;
   iEvent.getByLabel(m_dedxEstimatorTag,EstimatorH);
   const ValueMap<DeDxData> Estimator = *EstimatorH.product();

   edm::Handle<DeDxDataValueMap> DiscriminatorH;
   iEvent.getByLabel(m_dedxDiscriminatorTag,DiscriminatorH);
   const ValueMap<DeDxData> Discriminator = *DiscriminatorH.product();

   reco::TrackRef track = candidate.trackRef();
   candidate.setDedxEstimator    (Estimator    [track]);
   candidate.setDedxDiscriminator(Discriminator[track]);
}

