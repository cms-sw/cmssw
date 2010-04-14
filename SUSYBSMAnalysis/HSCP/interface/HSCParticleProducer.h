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
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCParticleProducer.h,v 1.1 2010/04/14 13:05:02 querten Exp $


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
using namespace susybsm;
class HSCParticleProducer : public edm::EDProducer {
  public:
    explicit HSCParticleProducer(const edm::ParameterSet&);
    ~HSCParticleProducer();

  private:
    virtual void beginJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    std::vector<HSCParticle> getHSCPSeedCollection(edm::Handle<reco::TrackCollection>& trackCollectionHandle,  edm::Handle<reco::MuonCollection>& muonCollectionHandle);

    // ----------member data ---------------------------
    edm::InputTag m_trackTag;
    edm::InputTag m_muonsTag;

    bool         useBetaFromTk;
    bool         useBetaFromMuon;
    bool         useBetaFromRpc;
    bool         useBetaFromEcal;

    float        minTkP;
    float        maxTkChi2;
    unsigned int minTkHits;
    float        minMuP;
    float        minDR;
    float        maxInvPtDiff;

    float        minTkdEdx;
    float        maxMuBeta;

    Beta_Calculator_TK*   beta_calculator_TK;
    Beta_Calculator_MUON* beta_calculator_MUON;
    Beta_Calculator_RPC*  beta_calculator_RPC;
    Beta_Calculator_ECAL* beta_calculator_ECAL;
};


