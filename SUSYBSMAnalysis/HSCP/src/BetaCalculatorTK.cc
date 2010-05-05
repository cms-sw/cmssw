#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorTK.h"

BetaCalculatorTK::BetaCalculatorTK(const edm::ParameterSet& iConfig){
  m_dedxEstimator1Tag     = iConfig.getParameter<edm::InputTag>("dedxEstimator1");
  m_dedxEstimator2Tag     = iConfig.getParameter<edm::InputTag>("dedxEstimator2");
  m_dedxEstimator3Tag     = iConfig.getParameter<edm::InputTag>("dedxEstimator3");
  m_dedxDiscriminator1Tag = iConfig.getParameter<edm::InputTag>("dedxDiscriminator1");
  m_dedxDiscriminator2Tag = iConfig.getParameter<edm::InputTag>("dedxDiscriminator2");
  m_dedxDiscriminator3Tag = iConfig.getParameter<edm::InputTag>("dedxDiscriminator3");
}


void BetaCalculatorTK::addInfoToCandidate(HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if(!candidate.hasTrackRef())return;

   edm::Handle<DeDxDataValueMap> Estimator1H;
   iEvent.getByLabel(m_dedxEstimator1Tag,Estimator1H);
   const ValueMap<DeDxData> Estimator1 = *Estimator1H.product();

   edm::Handle<DeDxDataValueMap> Estimator2H;
   iEvent.getByLabel(m_dedxEstimator2Tag,Estimator2H);
   const ValueMap<DeDxData> Estimator2 = *Estimator2H.product();

   edm::Handle<DeDxDataValueMap> Estimator3H;
   iEvent.getByLabel(m_dedxEstimator3Tag,Estimator3H);
   const ValueMap<DeDxData> Estimator3 = *Estimator3H.product();

   edm::Handle<DeDxDataValueMap> Discriminator1H;
   iEvent.getByLabel(m_dedxDiscriminator1Tag,Discriminator1H);
   const ValueMap<DeDxData> Discriminator1 = *Discriminator1H.product();

   edm::Handle<DeDxDataValueMap> Discriminator2H;
   iEvent.getByLabel(m_dedxDiscriminator2Tag,Discriminator2H);
   const ValueMap<DeDxData> Discriminator2 = *Discriminator2H.product();

   edm::Handle<DeDxDataValueMap> Discriminator3H;
   iEvent.getByLabel(m_dedxDiscriminator3Tag,Discriminator3H);
   const ValueMap<DeDxData> Discriminator3 = *Discriminator3H.product();

   reco::TrackRef track = candidate.trackRef();
   candidate.setDedxEstimator1    (Estimator1    [track]);
   candidate.setDedxEstimator2    (Estimator2    [track]);
   candidate.setDedxEstimator3    (Estimator3    [track]);
   candidate.setDedxDiscriminator1(Discriminator1[track]);
   candidate.setDedxDiscriminator2(Discriminator2[track]);
   candidate.setDedxDiscriminator3(Discriminator3[track]);
}

