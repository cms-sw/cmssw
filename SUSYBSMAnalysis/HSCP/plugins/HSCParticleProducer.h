// -*- C++ -*-
//
// Package:    HSCParticleProducer
// Class:      HSCParticleProducer
//
/**\class HSCParticleProducer HSCParticleProducer.h SUSYBSMAnalysis/HSCParticleProducer/interface/HSCParticleProducer.h

 Description: Producer for HSCP candidates, merging tracker dt information and rpc information

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont
//         Created:  Wed Oct 10 12:01:28 CEST 2007


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorTK.h"
#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorMUON.h"
#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorRPC.h"
#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorECAL.h"
#include "SUSYBSMAnalysis/HSCP/interface/CandidateSelector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "Math/GenVector/VectorUtil.h"

#include <TNtuple.h>
#include <TF1.h>

#include <vector>
#include <iostream>

//
// class decleration
//
class HSCParticleProducer : public edm::EDFilter {
  public:
    explicit HSCParticleProducer(const edm::ParameterSet&);
    ~HSCParticleProducer() override;

  private:
    void beginJob() override ;
    bool filter(edm::Event&, const edm::EventSetup&) override;
    void endJob() override ;

    std::vector<susybsm::HSCParticle> getHSCPSeedCollection(edm::Handle<reco::TrackCollection>& trackCollectionHandle,  edm::Handle<reco::MuonCollection>& muonCollectionHandle, edm::Handle<reco::MuonCollection>& MTmuonCollectionHandle);

    // ----------member data ---------------------------
    bool          Filter_;

    edm::EDGetTokenT<reco::TrackCollection> m_trackToken;
    edm::EDGetTokenT<reco::TrackCollection> m_trackIsoToken;
    edm::EDGetTokenT<reco::MuonCollection> m_muonsToken;
    edm::EDGetTokenT<reco::MuonCollection> m_MTmuonsToken;

    bool         useBetaFromTk;
    bool         useBetaFromMuon;
    bool         useBetaFromRpc;
    bool         useBetaFromEcal;

    float        minTkP;
    float        maxTkChi2;
    unsigned int minTkHits;
    float        minMuP;
    float        minSAMuPt;
    float        minMTMuPt;
    float        minDR;
    float        minMTDR;
    float        maxInvPtDiff;

    BetaCalculatorTK*   beta_calculator_TK;
    BetaCalculatorMUON* beta_calculator_MUON;
    BetaCalculatorRPC*  beta_calculator_RPC;
    BetaCalculatorECAL* beta_calculator_ECAL;

    std::vector<CandidateSelector*> Selectors;
};


