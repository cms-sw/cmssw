// -*- C++ -*-
//
// Package:    HSCP_Producer
// Class:      HSCP_Producer
// 
/**\class HSCP_Producer HSCP_Producer.h SUSYBSMAnalysis/HSCP_Producer/interface/HSCP_Producer.h

 Description: Producer for HSCP candidates, merging tracker dt information and rpc information

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCP_Producer.cc,v 1.10 2009/08/25 10:16:07 carrillo Exp $


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

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SUSYBSMAnalysis/HSCP/interface/Beta_Calculator_RPC.h"
#include "SUSYBSMAnalysis/HSCP/interface/Beta_Calculator_ECAL.h"
#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"

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
class HSCP_Producer : public edm::EDProducer {
  public:
    explicit HSCP_Producer(const edm::ParameterSet&);
    ~HSCP_Producer();

  private:
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    // ----------member data ---------------------------
    edm::InputTag m_trackTag;
    edm::InputTag m_trackDeDxEstimatorTag;
    edm::InputTag m_muonsTag;
    edm::InputTag m_muonsTOFTag;
    std::vector<HSCParticle> associate(susybsm::DeDxBetaCollection& ,const MuonTOFCollection&);
    void addBetaFromRPC(HSCParticle&);
    void addBetaFromEcal(HSCParticle&, edm::Handle<reco::TrackCollection>&, edm::Event&, const edm::EventSetup&);
    float minTkP, minDtP, maxTkBeta, minDR, maxInvPtDiff, maxChi2;
    unsigned int minTkHits, minTkMeas;

    Beta_Calculator_RPC*  beta_calculator_RPC;
    Beta_Calculator_ECAL* beta_calculator_ECAL;
};


