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

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"


#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

using namespace susybsm;

class  BetaCalculatorECAL{
   public:
      BetaCalculatorECAL(const edm::ParameterSet& iConfig);
      void  addInfoToCandidate(HSCParticle& candidate, edm::Handle<reco::TrackCollection>& tracks, edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
       TrackDetectorAssociator trackAssociator_; 
       TrackAssociatorParameters parameters_; 
};


