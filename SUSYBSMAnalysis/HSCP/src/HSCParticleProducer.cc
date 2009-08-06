// -*- C++ -*-
//
// Package:    HSCParticleProducer
// Class:      HSCParticleProducer
// 
/**\class HSCParticleProducer HSCParticleProducer.cc SUSYBSMAnalysis/HSCParticleProducer/src/HSCParticleProducer.cc

 Description: Producer for HSCP candidates, merging tracker dt information and rpc information

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
// Reworked and Ported to CMSSW_3_0_0 by Christophe Delaere
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCParticleProducer.cc,v 1.8 2009/05/13 21:55:47 delaer Exp $
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
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h> 
#include <Geometry/RPCGeometry/interface/RPCRoll.h> 

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
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
    std::vector<HSCParticle> associate(susybsm::DeDxBetaCollection& ,const MuonTOFCollection&);
    void addBetaFromRPC(HSCParticle&);
    void addBetaFromEcal(HSCParticle&, edm::Handle<reco::TrackCollection>&, edm::Event&, const edm::EventSetup&);
    float minTkP, minDtP, maxTkBeta, minDR, maxInvPtDiff, maxChi2;
    unsigned int minTkHits, minTkMeas;
    edm::ESHandle<RPCGeometry> rpcGeo;
    TrackDetectorAssociator trackAssociator_; 
    TrackAssociatorParameters parameters_; 
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
  parameters_.loadParameters( iConfig ); 

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

  // compute the RPC contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    addBetaFromRPC(*hscpcandidate);
  }

  // add the ecal contribution
  for(susybsm::HSCParticleCollection::iterator hscpcandidate = hscp->begin(); hscpcandidate < hscp->end(); ++hscpcandidate) {
    addBetaFromEcal(*hscpcandidate,trackCollectionHandle,iEvent,iSetup);
  }
  // output result
  iEvent.put(result); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCParticleProducer::beginJob(const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);
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

void HSCParticleProducer::addBetaFromRPC(HSCParticle& candidate) {
  // here we do basically as in RPCHSCPCANDIDATE.cc, but just for the hits on the muon of interest
  RPCBetaMeasurement result;
  // so, loop on the RPC hits of the muon
  trackingRecHit_iterator start,stop;
  if(candidate.hasMuonCombinedTrack()) {
    start = candidate.combinedTrack().recHitsBegin();
    stop  = candidate.combinedTrack().recHitsEnd();
  } else if(candidate.hasMuonStaTrack()) {
    start = candidate.staTrack().recHitsBegin();
    stop  = candidate.staTrack().recHitsEnd();
  } else return;
  for(trackingRecHit_iterator recHit = start; recHit != stop; ++recHit) {
    if ( (*recHit)->geographicalId().subdetId() != MuonSubdetId::RPC ) continue;
    if (!(*recHit)->isValid()) continue;
    RPCDetId rollId = (RPCDetId)(*recHit)->geographicalId();
    RPCGeomServ rpcsrv(rollId);
    LocalPoint recHitPos=(*recHit)->localPosition();
    const RPCRoll* rollasociated = rpcGeo->roll(rollId);
    const BoundPlane & RPCSurface = rollasociated->surface();
    RPCHit4D ThisHit;
    ThisHit.bx = ((RPCRecHit*)(&(**recHit)))->BunchX();
    ThisHit.gp = RPCSurface.toGlobal(recHitPos);
    ThisHit.id = (RPCDetId)(*recHit)->geographicalId().rawId();
    result.hits.push_back(ThisHit);
  }
  // here we go on with the RPC procedure 
  std::sort(result.hits.begin(), result.hits.end());
  int lastbx=7;
  bool decreasing = true;
  bool outOfTime = false;
  for(std::vector<RPCHit4D>::iterator point = result.hits.begin(); point < result.hits.end(); ++point) {
    outOfTime |= (point->bx!=0); //condition 1: at least one measurement must have BX!=0
    decreasing &= (point->bx<=lastbx); //condition 2: BX must be decreasing when going inside-out.
    lastbx = point->bx;
  }
  result.isCandidate = (outOfTime&&decreasing);
  result.beta = 1; // here we should get some pattern-based estimate
  candidate.setRpc(result);
}

void HSCParticleProducer::addBetaFromEcal(HSCParticle& candidate, edm::Handle<reco::TrackCollection>& tracks, 
                                          edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // the calo info object
  CaloBetaMeasurement result;
  
  // select the track
  reco::Track track;
  if(candidate.hasMuonCombinedTrack()) {
    track = candidate.combinedTrack();
  } else if(candidate.hasTrackerTrack()) {
    track = candidate.trackerTrack();
  } else if(candidate.hasMuonStaTrack()) {
    track = candidate.staTrack();
  } else return;
  
  // compute the track isolation
  result.trkisodr=100;
  for(reco::TrackCollection::const_iterator ndTrack = tracks->begin(); ndTrack != tracks->end(); ++ndTrack) {
      double dr=sqrt(pow((track.outerEta()-ndTrack->outerEta()),2)+pow((track.outerPhi()-ndTrack->outerPhi()),2));
      if(dr>0.00001 && dr<result.trkisodr) result.trkisodr=dr;
  }

  // use the track associator to propagate to the calo
  TrackDetMatchInfo info = trackAssociator_.associate( iEvent, iSetup, 
                                                       trackAssociator_.getFreeTrajectoryState(iSetup, track),
                                                       parameters_ );

  // extract various quantities
  result.ecalenergy = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
  DetId centerId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
  GlobalPoint position = info.getPosition(centerId);
  double matchedR = sqrt(pow(position.x(),2)+pow(position.y(),2)+pow(position.z(),2));
  result.ecal5by5dir = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
  for(std::vector<const EcalRecHit*>::const_iterator hit = info.crossedEcalRecHits.begin(); 
      hit != info.crossedEcalRecHits.end(); ++hit) {
    result.ecaltime += (*hit)->time();	
  }
  if(info.crossedEcalRecHits.size()) {
    result.ecaltime /= info.crossedEcalRecHits.size();
    result.ecalbeta = (matchedR+23)/(result.ecaltime*25.*30+matchedR);
  }
  result.hcalenergy = info.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
  result.hoenergy = info.crossedEnergy(TrackDetMatchInfo::HORecHits);
  centerId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
  result.hcal3by3dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
  result.hcal5by5dir = info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);
  // conclude by putting all that in the candidate
  candidate.setCalo(result);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCParticleProducer);
