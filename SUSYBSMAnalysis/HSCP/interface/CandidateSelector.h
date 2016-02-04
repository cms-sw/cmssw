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

#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

class  CandidateSelector{
   public:
      CandidateSelector(const edm::ParameterSet& iConfig);
      bool isSelected(susybsm::HSCParticle& candidate);

      bool  isTrack;
      bool  isMuon;
      bool  isMuonSTA;
      bool  isMuonGB;
      bool  isMuonTK;
      bool  isRpc;
      bool  isEcal;

      int   minTrackHits;
      float minTrackP;
      float minTrackPt;
      float minDedx;

      float minMuonP;
      float minMuonPt;      

      float maxMuTimeDtBeta;
      float minMuTimeDtNdof;
      float maxMuTimeCscBeta;
      float minMuTimeCscNdof;
      float maxMuTimeCombinedBeta;
      float minMuTimeCombinedNdof;

      float maxBetaRpc;
      float maxBetaEcal;
};


