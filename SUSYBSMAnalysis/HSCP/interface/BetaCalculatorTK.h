// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"


#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"


class  BetaCalculatorTK{
   public:
      BetaCalculatorTK(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
      void  addInfoToCandidate(susybsm::HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup);

      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator1Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator2Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator3Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator4Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator5Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxEstimator6Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator1Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator2Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator3Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator4Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator5Token;
      edm::EDGetTokenT<reco::DeDxDataValueMap> m_dedxDiscriminator6Token;
};


