// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"


#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"


class  BetaCalculatorTK{
   public:
      BetaCalculatorTK(const edm::ParameterSet& iConfig);
      void  addInfoToCandidate(susybsm::HSCParticle& candidate, edm::Event& iEvent, const edm::EventSetup& iSetup);

      edm::InputTag m_dedxEstimator1Tag;
      edm::InputTag m_dedxEstimator2Tag;
      edm::InputTag m_dedxEstimator3Tag;
      edm::InputTag m_dedxEstimator4Tag;
      edm::InputTag m_dedxEstimator5Tag;
      edm::InputTag m_dedxEstimator6Tag;
      edm::InputTag m_dedxDiscriminator1Tag;
      edm::InputTag m_dedxDiscriminator2Tag;
      edm::InputTag m_dedxDiscriminator3Tag;
      edm::InputTag m_dedxDiscriminator4Tag;
      edm::InputTag m_dedxDiscriminator5Tag;
      edm::InputTag m_dedxDiscriminator6Tag;
};


