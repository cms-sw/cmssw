#include "SUSYBSMAnalysis/HSCP/interface/BetaCalculatorRPC.h"

using namespace susybsm;


BetaCalculatorRPC::BetaCalculatorRPC(const edm::ParameterSet& iConfig){

  rpcRecHitsLabel = iConfig.getParameter<edm::InputTag>("rpcRecHits");

}

void BetaCalculatorRPC::algo(const std::vector<susybsm::RPCHit4D>& uHSCPRPCRecHits){
  std::vector<susybsm::RPCHit4D> HSCPRPCRecHits = uHSCPRPCRecHits;
  int lastbx=-7;
  bool outOfTime = false;
  bool increasing = true;
  bool anydifferentzero = true;
  bool anydifferentone = true;
  
  //std::cout<<"Inside BetaCalculatorRPC \t \t Preliminar loop on the RPCHit4D!!!"<<std::endl;

  std::sort(HSCPRPCRecHits.begin(), HSCPRPCRecHits.end()); //Organizing them

  for(std::vector<susybsm::RPCHit4D>::iterator point = HSCPRPCRecHits.begin(); point < HSCPRPCRecHits.end(); ++point) {
    outOfTime |= (point->bx!=0); //condition 1: at least one measurement must have BX!=0
    increasing &= (point->bx>=lastbx); //condition 2: BX must be increase when going inside-out.
    anydifferentzero &= (!point->bx==0); //to check one knee withoutzeros
    anydifferentone &= (!point->bx==1); //to check one knee withoutones
    lastbx = point->bx;
    //float r=point->gp.mag();
    //std::cout<<"Inside BetaCalculatorRPC \t \t  r="<<r<<" phi="<<point->gp.phi()<<" eta="<<point->gp.eta()<<" bx="<<point->bx<<" outOfTime"<<outOfTime<<" increasing"<<increasing<<" anydifferentzero"<<anydifferentzero<<std::endl;
  }
  
  bool Candidate = (outOfTime&&increasing);

  // here we should get some pattern-based estimate

  //Counting knees

  float delay=12.5;
  lastbx=-7; //already declared for the preliminar loop
  int knees=0;
  float maginknee = 0;
  float maginfirstknee = 0;
  for(std::vector<susybsm::RPCHit4D>::iterator point = HSCPRPCRecHits.begin(); point < HSCPRPCRecHits.end(); ++point) {
    if(lastbx==-7){
      maginfirstknee = point->gp.mag();
    }else if((lastbx!=point->bx)){
      //std::cout<<"Inside BetaCalculatorRPC \t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
      maginknee=point->gp.mag();
      knees++;
    }
    lastbx=point->bx;
  }
      
  if(knees==0){
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t knees="<<knees<<std::endl;
    betavalue=maginfirstknee/(25.-delay+maginfirstknee/30.)/30.;
  }else if(knees==1){
    float betavalue1=0;
    float betavalue2=0;
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t knees="<<knees<<std::endl;
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t anydifferentzero="<<anydifferentzero<<" anydifferentone="<<anydifferentone<<std::endl;
    if(!anydifferentzero){
      betavalue=maginknee/(25-delay+maginknee/30.)/30.;
    }else if(!anydifferentone){//i.e non zeros and no ones
      betavalue=maginknee/(50-delay+maginknee/30.)/30.;
    }else{
      betavalue1=maginknee/(25-delay+maginknee/30.)/30.;
      float dr =(maginknee-maginfirstknee);
      betavalue2 = dr/(25.-delay+dr/30.);
      //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t \t not zero neither ones betavalue1="<<betavalue1<<" betavalue2="<<betavalue2<<std::endl;
      betavalue = (betavalue1 + betavalue2)*0.5;
    }
  }else if(knees==2){
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t knees="<<knees<<std::endl;
    knees=0;
    float betavalue1=0;
    float betavalue2=0;
    lastbx=-7;
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t looping again on the RPCRecHits4D="<<knees<<std::endl;
    for(std::vector<susybsm::RPCHit4D>::iterator point = HSCPRPCRecHits.begin(); point < HSCPRPCRecHits.end(); ++point) {
      if(lastbx==-7){
	maginfirstknee = point->gp.mag();
      }else if((lastbx!=point->bx)){
	//std::cout<<"Inside BetaCalculatorRPC \t \t \t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
	knees++;
	if(knees==2){
	  float maginsecondknee=point->gp.mag();
	  betavalue1=maginknee/(25-delay+maginknee/30.)/30.;
	  float dr =(maginknee-maginsecondknee);
	  betavalue2 = dr/(25.+dr/30.);
	  //std::cout<<"Inside BetaCalculatorRPC \t \t \t \t \t betavalue1="<<betavalue1<<" betavalue2="<<betavalue2<<std::endl;
	}
      }
      lastbx=point->bx;
    }
    betavalue = (betavalue1 + betavalue2)*0.5;
  }
  
  if(Candidate){
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t yes! We found an HSCPs let's try to estimate beta"<<std::endl;
  }else{
    //std::cout<<"Inside BetaCalculatorRPC \t \t \t seems that there is no RPC HSCP Candidate in the set of RPC4DHit"<<std::endl;
    betavalue = 1.;
  }
  
  if(HSCPRPCRecHits.size()==0){
    //std::cout<<"Inside BetaCalculatorRPC \t WARINNG EMPTY RPC4DRecHits CONTAINER!!!"<<std::endl;
    betavalue = 1.;
  }
}



void BetaCalculatorRPC::addInfoToCandidate(HSCParticle& candidate, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(rpcRecHitsLabel,rpcHits);


  // here we do basically as in RPCHSCPCANDIDATE.cc, but just for the hits on the muon of interest
  RPCBetaMeasurement result;
  std::vector<RPCHit4D> hits;
  // so, loop on the RPC hits of the muon
  trackingRecHit_iterator start,stop;
  reco::Track track;

  if(      candidate.hasMuonRef() && candidate.muonRef()->combinedMuon()  .isNonnull()){ 
     start = candidate.muonRef()->combinedMuon()->recHitsBegin();
     stop  = candidate.muonRef()->combinedMuon()->recHitsEnd();    
  }else if(candidate.hasMuonRef() && candidate.muonRef()->standAloneMuon().isNonnull()){ track=*(candidate.muonRef()->standAloneMuon());
     start = candidate.muonRef()->standAloneMuon()->recHitsBegin();
     stop  = candidate.muonRef()->standAloneMuon()->recHitsEnd();  
  }else return;
/*
  if(candidate.hasMuonCombinedTrack()) {
    start = candidate.combinedTrack().recHitsBegin();
    stop  = candidate.combinedTrack().recHitsEnd();
  } else if(candidate.hasMuonStaTrack()) {
    start = candidate.staTrack().recHitsBegin();
    stop  = candidate.staTrack().recHitsEnd();
  } else return; 
*/
  for(trackingRecHit_iterator recHit = start; recHit != stop; ++recHit) {
    if ( (*recHit)->geographicalId().subdetId() != MuonSubdetId::RPC ) continue;
    if ( (*recHit)->geographicalId().det() != DetId::Muon  ) continue;
    if (!(*recHit)->isValid()) continue; //Is Valid?
       
    RPCDetId rollId = (RPCDetId)(*recHit)->geographicalId();

    typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
    rangeRecHits recHitCollection =  rpcHits->get(rollId);
    RPCRecHitCollection::const_iterator recHitC;
    int size = 0;
    int clusterS=0;
    for(recHitC = recHitCollection.first; recHitC != recHitCollection.second ; recHitC++) {
      clusterS=(*recHitC).clusterSize();
//      RPCDetId rollId = (RPCDetId)(*recHitC).geographicalId();
//      std::cout<<"\t \t \t \t"<<rollId<<" bx "<<(*recHitC).BunchX()<<std::endl;
      size++;
    }
    if(size>1) continue; //Is the only RecHit in this roll.?                                                                                                         
    if(clusterS>4) continue; //Is the Cluster Size 5 or bigger?    
    
    LocalPoint recHitPos=(*recHit)->localPosition();
    const RPCRoll* rollasociated = rpcGeo->roll(rollId);
    const BoundPlane & RPCSurface = rollasociated->surface();
    
    RPCHit4D ThisHit;
    ThisHit.bx = ((RPCRecHit*)(&(**recHit)))->BunchX();
    ThisHit.gp = RPCSurface.toGlobal(recHitPos);
    ThisHit.id = (RPCDetId)(*recHit)->geographicalId().rawId();
    hits.push_back(ThisHit);
    
  }
  // here we go on with the RPC procedure 
  std::sort(hits.begin(), hits.end());
  int lastbx=-7;
  bool increasing = true;
  bool outOfTime = false;
  for(std::vector<RPCHit4D>::iterator point = hits.begin(); point < hits.end(); ++point) {
    outOfTime |= (point->bx!=0); //condition 1: at least one measurement must have BX!=0
    increasing &= (point->bx>=lastbx); //condition 2: BX must increase when going inside-out.
    lastbx = point->bx;
  }
  result.isCandidate = (outOfTime&&increasing);
 
  //result.beta = 1; // here we should get some pattern-based estimate
  algo(hits);
  result.beta = beta();
  candidate.setRpc(result);
}


