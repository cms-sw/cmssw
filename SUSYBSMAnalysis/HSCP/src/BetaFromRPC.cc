#include "SUSYBSMAnalysis/HSCP/interface/BetaFromRPC.h"

BetaFromRPC::BetaFromRPC(std::vector<susybsm::RPCHit4D> HSCPRPCRecHits){

  int lastbx=-7;
  bool outOfTime = false;
  bool increasing = true;
  bool anydifferentzero = true;
  bool anydifferentone = true;
  
  //std::cout<<"Inside BetaFromRPC \t \t Preliminar loop on the RPCHit4D!!!"<<std::endl;

  std::sort(HSCPRPCRecHits.begin(), HSCPRPCRecHits.end()); //Organizing them

  for(std::vector<susybsm::RPCHit4D>::iterator point = HSCPRPCRecHits.begin(); point < HSCPRPCRecHits.end(); ++point) {
    float r=point->gp.mag();
    outOfTime |= (point->bx!=0); //condition 1: at least one measurement must have BX!=0
    increasing &= (point->bx>=lastbx); //condition 2: BX must be increase when going inside-out.
    anydifferentzero &= (!point->bx==0); //to check one knee withoutzeros
    anydifferentone &= (!point->bx==1); //to check one knee withoutones
    lastbx = point->bx;
    //std::cout<<"Inside BetaFromRPC \t \t  r="<<r<<" phi="<<point->gp.phi()<<" eta="<<point->gp.eta()<<" bx="<<point->bx<<" outOfTime"<<outOfTime<<" increasing"<<increasing<<" anydifferentzero"<<anydifferentzero<<std::endl;
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
      //std::cout<<"Inside BetaFromRPC \t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
      maginknee=point->gp.mag();
      knees++;
    }
    lastbx=point->bx;
  }
      
  if(knees==0){
    //std::cout<<"Inside BetaFromRPC \t \t \t \t knees="<<knees<<std::endl;
    betavalue=maginfirstknee/(25.-delay+maginfirstknee/30.)/30.;
  }else if(knees==1){
    float betavalue1=0;
    float betavalue2=0;
    //std::cout<<"Inside BetaFromRPC \t \t \t \t knees="<<knees<<std::endl;
    //std::cout<<"Inside BetaFromRPC \t \t \t \t anydifferentzero="<<anydifferentzero<<" anydifferentone="<<anydifferentone<<std::endl;
    if(!anydifferentzero){
      betavalue=maginknee/(25-delay+maginknee/30.)/30.;
    }else if(!anydifferentone){//i.e non zeros and no ones
      betavalue=maginknee/(50-delay+maginknee/30.)/30.;
    }else{
      betavalue1=maginknee/(25-delay+maginknee/30.)/30.;
      float dr =(maginknee-maginfirstknee);
      betavalue2 = dr/(25.-delay+dr/30.);
      //std::cout<<"Inside BetaFromRPC \t \t \t \t \t not zero neither ones betavalue1="<<betavalue1<<" betavalue2="<<betavalue2<<std::endl;
      betavalue = (betavalue1 + betavalue2)*0.5;
    }
  }else if(knees==2){
    //std::cout<<"Inside BetaFromRPC \t \t \t \t knees="<<knees<<std::endl;
    knees=0;
    float betavalue1=0;
    float betavalue2=0;
    lastbx=-7;
    //std::cout<<"Inside BetaFromRPC \t \t \t \t looping again on the RPCRecHits4D="<<knees<<std::endl;
    for(std::vector<susybsm::RPCHit4D>::iterator point = HSCPRPCRecHits.begin(); point < HSCPRPCRecHits.end(); ++point) {
      if(lastbx==-7){
	maginfirstknee = point->gp.mag();
      }else if((lastbx!=point->bx)){
	//std::cout<<"Inside BetaFromRPC \t \t \t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
	knees++;
	if(knees==2){
	  float maginsecondknee=point->gp.mag();
	  betavalue1=maginknee/(25-delay+maginknee/30.)/30.;
	  float dr =(maginknee-maginsecondknee);
	  betavalue2 = dr/(25.+dr/30.);
	  //std::cout<<"Inside BetaFromRPC \t \t \t \t \t betavalue1="<<betavalue1<<" betavalue2="<<betavalue2<<std::endl;
	}
      }
      lastbx=point->bx;
    }
    betavalue = (betavalue1 + betavalue2)*0.5;
  }
  
  if(Candidate){
    //std::cout<<"Inside BetaFromRPC \t \t \t yes! We found an HSCPs let's try to estimate beta"<<std::endl;
  }else{
    //std::cout<<"Inside BetaFromRPC \t \t \t seems that there is no RPC HSCP Candidate in the set of RPC4DHit"<<std::endl;
    betavalue = 1.;
  }
  
  if(HSCPRPCRecHits.size()==0){
    //std::cout<<"Inside BetaFromRPC \t WARINNG EMPTY RPC4DRecHits CONTAINER!!!"<<std::endl;
    betavalue = 1.;
  }
}
