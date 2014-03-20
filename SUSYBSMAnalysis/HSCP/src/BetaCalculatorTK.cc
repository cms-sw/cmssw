#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorTK.h"

using namespace edm;
using namespace reco;
using namespace susybsm;


BetaCalculatorTK::BetaCalculatorTK(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC){
/*
  m_dedxEstimator1Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator1"));
  m_dedxEstimator2Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator2"));
  m_dedxEstimator3Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator3"));
  m_dedxEstimator4Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator4"));
  m_dedxEstimator5Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator5"));
  m_dedxEstimator6Token     = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxEstimator6"));
  m_dedxDiscriminator1Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator1"));
  m_dedxDiscriminator2Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator2"));
  m_dedxDiscriminator3Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator3"));
  m_dedxDiscriminator4Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator4"));
  m_dedxDiscriminator5Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator5"));
  m_dedxDiscriminator6Token = consumes<DeDxDataValueMap>(iConfig.getParameter<edm::InputTag>("dedxDiscriminator6"));
*/
}


void BetaCalculatorTK::addInfoToCandidate(HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   //Do nothing since all dE/dx object are external and get be accessed via reference
   return;
/*
   if(!candidate.hasTrackRef())return;

   edm::Handle<DeDxDataValueMap> Estimator1H;
   iEvent.getByToken(m_dedxEstimator1Token,Estimator1H);
   const ValueMap<DeDxData> Estimator1 = *Estimator1H.product();

   edm::Handle<DeDxDataValueMap> Estimator2H;
   iEvent.getByToken(m_dedxEstimator2Token,Estimator2H);
   const ValueMap<DeDxData> Estimator2 = *Estimator2H.product();

   edm::Handle<DeDxDataValueMap> Estimator3H;
   iEvent.getByToken(m_dedxEstimator3Token,Estimator3H);
   const ValueMap<DeDxData> Estimator3 = *Estimator3H.product();

   edm::Handle<DeDxDataValueMap> Estimator4H;
   iEvent.getByToken(m_dedxEstimator4Token,Estimator4H);
   const ValueMap<DeDxData> Estimator4 = *Estimator4H.product();

   edm::Handle<DeDxDataValueMap> Estimator5H;
   iEvent.getByToken(m_dedxEstimator5Token,Estimator5H);
   const ValueMap<DeDxData> Estimator5 = *Estimator5H.product();

   edm::Handle<DeDxDataValueMap> Estimator6H;
   iEvent.getByToken(m_dedxEstimator6Token,Estimator6H);
   const ValueMap<DeDxData> Estimator6 = *Estimator6H.product();

   edm::Handle<DeDxDataValueMap> Discriminator1H;
   iEvent.getByToken(m_dedxDiscriminator1Token,Discriminator1H);
   const ValueMap<DeDxData> Discriminator1 = *Discriminator1H.product();

   edm::Handle<DeDxDataValueMap> Discriminator2H;
   iEvent.getByToken(m_dedxDiscriminator2Token,Discriminator2H);
   const ValueMap<DeDxData> Discriminator2 = *Discriminator2H.product();

   edm::Handle<DeDxDataValueMap> Discriminator3H;
   iEvent.getByToken(m_dedxDiscriminator3Token,Discriminator3H);
   const ValueMap<DeDxData> Discriminator3 = *Discriminator3H.product();

   edm::Handle<DeDxDataValueMap> Discriminator4H;
   iEvent.getByToken(m_dedxDiscriminator4Token,Discriminator4H);
   const ValueMap<DeDxData> Discriminator4 = *Discriminator4H.product();

   edm::Handle<DeDxDataValueMap> Discriminator5H;
   iEvent.getByToken(m_dedxDiscriminator5Token,Discriminator5H);
   const ValueMap<DeDxData> Discriminator5 = *Discriminator5H.product();

   edm::Handle<DeDxDataValueMap> Discriminator6H;
   iEvent.getByToken(m_dedxDiscriminator6Token,Discriminator6H);
   const ValueMap<DeDxData> Discriminator6 = *Discriminator6H.product();

   reco::TrackRef track = candidate.trackRef();
   candidate.setDedxEstimator1    (Estimator1    [track]);
   candidate.setDedxEstimator2    (Estimator2    [track]);
   candidate.setDedxEstimator3    (Estimator3    [track]);
   candidate.setDedxEstimator4    (Estimator4    [track]);
   candidate.setDedxEstimator5    (Estimator5    [track]);
   candidate.setDedxEstimator6    (Estimator6    [track]);
   candidate.setDedxDiscriminator1(Discriminator1[track]);
   candidate.setDedxDiscriminator2(Discriminator2[track]);
   candidate.setDedxDiscriminator3(Discriminator3[track]);
   candidate.setDedxDiscriminator4(Discriminator4[track]);
   candidate.setDedxDiscriminator5(Discriminator5[track]);
   candidate.setDedxDiscriminator6(Discriminator6[track]);
*/
}

