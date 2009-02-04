// -*- C++ -*-
//
// Package:    HSCParticleProducer
// Class:      HSCParticleProducer
// 
/**\class HSCParticleProducer HSCParticleProducer.cc SUSYBSMAnalysis/HSCParticleProducer/src/HSCParticleProducer.cc

 Description: Producer for HSCP candidates, merging tracker and dt information

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCParticleProducer.cc,v 1.5 2008/08/26 14:09:25 arizzi Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
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
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    // ----------member data ---------------------------
    edm::InputTag m_trackTag;
    edm::InputTag m_trackDeDxEstimatorTag;
    edm::InputTag m_muonsTag;
    edm::InputTag m_muonsTOFTag;
    std::vector<HSCParticle> associate( susybsm::DeDxBetaCollection & tk ,const MuonTOFCollection & dts );
    float minTkP, minDtP, maxTkBeta, minDR, maxInvPtDiff, maxChi2;
    unsigned int minTkHits, minTkMeas;
};

HSCParticleProducer::HSCParticleProducer(const edm::ParameterSet& iConfig) {
  using namespace edm;
  using namespace std;

  // the input collections
  m_trackDeDxEstimatorTag = iConfig.getParameter<edm::InputTag>("trackDeDxEstimator");
  m_trackTag = iConfig.getParameter<edm::InputTag>("tracks");
  m_muonsTag = iConfig.getParameter<edm::InputTag>("muons");
  m_muonsTOFTag = iConfig.getParameter<edm::InputTag>("muonsTOF");

  // the parameters
  minTkP=iConfig.getParameter<double>("minTkP"); // 30
  maxTkBeta=iConfig.getParameter<double>("maxTkBeta"); //0.9;
  minDtP=iConfig.getParameter<double>("minDtP"); //30
  minDR=iConfig.getParameter<double>("minDR"); //0.1
  maxInvPtDiff=iConfig.getParameter<double>("maxInvPtDiff"); //0.005
  maxChi2=iConfig.getParameter<double>("maxTkChi2"); //5
  minTkHits=iConfig.getParameter<uint32_t>("minTkHits"); //9
  minTkMeas=iConfig.getParameter<uint32_t>("minTkMeas"); //9

  // what I produce
  produces<susybsm::HSCParticleCollection >();
}

HSCParticleProducer::~HSCParticleProducer() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HSCParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace susybsm;

  // information from the muon system: TOF
  Handle<MuonTOFCollection> betaRecoH;
  iEvent.getByLabel(m_muonsTOFTag,betaRecoH);
  const MuonTOFCollection & dtInfos  =  *betaRecoH.product();

  // information from the tracker: dE/dx
  Handle<DeDxDataValueMap> dedxH;
  iEvent.getByLabel(m_trackDeDxEstimatorTag,dedxH);
  const ValueMap<DeDxData> dEdxTrack = *dedxH.product();
  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_trackTag,trackCollectionHandle);
  DeDxBetaCollection   tkInfos;
  for(unsigned int i=0; i<trackCollectionHandle->size(); i++){
    reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );
    const DeDxData& dedx = dEdxTrack[track];
    if(track->normalizedChi2()     <  maxChi2   && 
       track->numberOfValidHits()  >= minTkHits && 
       dedx.numberOfMeasurements() >= minTkMeas    ) {
      float k=0.4;  //919/2.75*0.0012;
      //float k2=0.432; //919/2.55*0.0012;
      tkInfos.push_back(DeDxBeta(track,dedx,k));
    }
  }
  
  // creates the output collection
  susybsm::HSCParticleCollection * hscp = new susybsm::HSCParticleCollection; 
  std::auto_ptr<susybsm::HSCParticleCollection> result(hscp);
  
  // match TOF and dE/dx info
  *hscp=associate(tkInfos,dtInfos);

  // output result
  iEvent.put(result); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCParticleProducer::beginJob(const edm::EventSetup&) {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCParticleProducer::endJob() {
}

std::vector<HSCParticle> HSCParticleProducer::associate( DeDxBetaCollection & tks ,const MuonTOFCollection & dtsInput ) {
  
  // the output collection
  std::vector<HSCParticle> result;
  
  // make a local copy of the MuonTOFCollection so that we can remove elements 
  std::vector<MuonTOF> dts;
  std::copy (dtsInput.begin(),dtsInput.end(), std::back_inserter (dts));

  // loop on tracker measurements and try to associate a muon from the MuonTOFCollection
  LogDebug("matching") << "number of track measurements: " << tks.size();
  float minTkInvBeta2=1./(maxTkBeta*maxTkBeta);
  for(size_t i=0;i<tks.size();i++) {
    LogDebug("matching") << "Pt = " << tks[i].track()->pt() 
                         << "; 1/beta2 = " << tks[i].invBeta2();
    if( tks[i].track().isNonnull() && tks[i].track()->pt() > minTkP && tks[i].invBeta2() >= minTkInvBeta2 ) {
      LogDebug("matching") << "Found one candidate using the tracker.";
      float min=1000;
      int found = -1;
      for(size_t j=0;j<dts.size();j++) {
        if(dts[j].first->combinedMuon().isNonnull()) {
          float invDT=1./dts[j].first->combinedMuon()->pt();
          float invTK=1./tks[i].track()->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[j].first->combinedMuon()->momentum(), tks[i].track()->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
	  LogDebug("matching") << "Tracker candidate is associated to muon candidate " << j;
        }
      }
      HSCParticle candidate;
      candidate.setTk(tks[i]);
      if(found>=0) {
        candidate.setDt(dts[found]);
        dts.erase(dts.begin()+found);
      } else {
        LogDebug("matching") << "No associated muon candidate found at eta = " << tks[i].track()->eta();
      }
      result.push_back(candidate);
    }
  }

  // loop on the remaining muons and try to associate a trakcer measurement.
  // There should be none, unless two tracks match the same muon
  LogDebug("matching") << "number of remaining muon measurements: " << dts.size();
  for(size_t i=0;i<dts.size();i++) {
    if(dts[i].first->combinedMuon().isNonnull() && dts[i].first->combinedMuon()->pt() > minDtP ) {
      LogDebug("matching") << "Found one candidate using the dts.";
      float min=1000;
      int found = -1;
      for(size_t j=0;j<tks.size();j++) {
        if( tks[j].track().isNonnull() ) {
          float invDT=1./dts[i].first->combinedMuon()->pt();
          float invTK=1./tks[j].track()->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[i].first->combinedMuon()->momentum(), tks[j].track()->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
	  LogDebug("matching") << "Muon candidate is associated to tracker candidate " << j << ". At least two muons associated to the same track ?";
        }
      }
      HSCParticle candidate;
      candidate.setDt(dts[i]);
      if(found>=0 ) {
        candidate.setTk(tks[found]);
      }
      result.push_back(candidate);
    }
  }

  // returns the result
  LogDebug("matching") << "Matching between trakcer and dt information over.";
  return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCParticleProducer);
