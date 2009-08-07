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
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "SUSYBSMAnalysis/HSCP/interface/RPCHSCPCANDIDATE.h"

typedef struct {
  int id;
  int bx;
  GlobalPoint gp;
} RPC4DHit;

bool bigmag(const RPC4DHit &Point1,const RPC4DHit &Point2){
  if((Point2).gp.mag() > (Point1).gp.mag()) return true;
  else return false;
}

float RPCHSCPCANDIDATE::etarange(float eta1,float eta2,float eta3){
  float etamax = eta1; 
  if(eta2>etamax) etamax = eta2;
  if(eta3>etamax) etamax = eta3;
  
  float etamin = eta1;
  if(eta2<etamin) etamin = eta2;
  if(eta3<etamin) etamin = eta3;
  
  return fabs(etamax-etamin);
} 

float RPCHSCPCANDIDATE::dist(float phi1,float phi2){
  if(fabs(phi1-phi2)>3.14159265) return 2*3.1415926535-fabs(phi1-phi2);
  else return fabs(phi1-phi2);
}

float RPCHSCPCANDIDATE::dist3(float phi1,float phi2,float phi3){
  return dist(phi1,phi2)+dist(phi2,phi3)+dist(phi3,phi1);
}

RPCHSCPCANDIDATE::RPCHSCPCANDIDATE(edm::Event& iEvent,const edm::EventSetup& iSetup){

  using namespace edm;

   //int event = iEvent.id().event();

   Handle<RPCRecHitCollection> rpcRecHits; 
   iEvent.getByLabel("rpcRecHits","",rpcRecHits);

   edm::ESHandle<RPCGeometry> rpcGeo;
   iSetup.get<MuonGeometryRecord>().get(rpcGeo);

   int nRPC = 0;


   RPCRecHitCollection::const_iterator recHit;
      
   for (recHit = rpcRecHits->begin(); recHit != rpcRecHits->end(); recHit++) {
     //    RPCDetId id = (RPCDetId)(*recHit).rpcId();
     //    LocalPoint rhitlocal = (*recHit).localPosition();
     nRPC++;
   }


   std::cout<<"Inside the class "<<"The Number of Rec Hits is "<<nRPC<<std::endl;

   std::vector<RPC4DHit> BestAngularMatch;
   std::vector<RPC4DHit> bufferBestAngularMatch;
   std::vector<RPC4DHit> GlobalRPC4DHits;
   std::vector<RPC4DHit> GlobalRPC4DHitsNoBx0;

   int k=0;
   
   for (recHit = rpcRecHits->begin(); recHit != rpcRecHits->end(); recHit++) {
     RPCDetId rollId = (RPCDetId)(*recHit).rpcId();
     RPCGeomServ rpcsrv(rollId);
     LocalPoint recHitPos=recHit->localPosition();    
     const RPCRoll* rollasociated = rpcGeo->roll(rollId);
     const BoundPlane & RPCSurface = rollasociated->surface(); 
     GlobalPoint RecHitInGlobal = RPCSurface.toGlobal(recHitPos);
     
     int BX = recHit->BunchX();
     std::cout<<"Inside the class "<<"\t \t We have an RPC Rec Hit! bx="<<BX<<" Roll="<<rpcsrv.name()<<" Global Position="<<RecHitInGlobal<<std::endl;
     
     RPC4DHit ThisHit;
     ThisHit.bx =  BX;
     ThisHit.gp = RecHitInGlobal;
     ThisHit.id = k;
     GlobalRPC4DHits.push_back(ThisHit);
     if(BX!=0)GlobalRPC4DHitsNoBx0.push_back(ThisHit);
     k++;
   }
   
   assert(nRPC==int(GlobalRPC4DHits.size()));

   if(GlobalRPC4DHitsNoBx0.size()==0){
     std::cout<<"Inside the class "<<"No Chances for HSCPs"<<std::endl;
   }   
   
   float minangularspread = 100.;
   float minetaspread = 100.;
   
   for(std::vector<RPC4DHit>::iterator Point = GlobalRPC4DHitsNoBx0.begin(); Point!=GlobalRPC4DHitsNoBx0.end(); ++Point){ 
     for(std::vector<RPC4DHit>::iterator Point2 = GlobalRPC4DHits.begin(); Point2!=GlobalRPC4DHits.end(); ++Point2){if(Point2->id!=Point->id){
       for(std::vector<RPC4DHit>::iterator Point3 = Point2; Point3!=GlobalRPC4DHits.end(); ++Point3){if(Point3->id!=Point2->id && Point3->id!=Point->id){
	 float angularspread = dist3(Point->gp.phi().value(),Point2->gp.phi().value(),Point3->gp.phi().value()); 
	 if(angularspread<minangularspread){
	   minangularspread = angularspread;
	   std::cout<<"Inside the class "<<"For phi"<< Point->gp.phi()<<" "<<Point2->gp.phi()<<" "<<Point3->gp.phi()<<std::endl;
	   phivalue=(Point->gp.phi().value()+Point2->gp.phi().value()+Point3->gp.phi().value())/3.; //implementar average
	   
	   float etaspread = etarange(float(Point->gp.eta()),float(Point2->gp.eta()),float(Point3->gp.eta()));
	   if(etaspread<0.2){
	     minetaspread = etaspread;
	     std::cout<<"Inside the class "<<"For eta"<<Point->gp.eta()<<" "<<Point2->gp.eta()<<" "<<Point3->gp.eta()<<std::endl;
	     etavalue = (float(Point->gp.eta())+float(Point2->gp.eta())+float(Point3->gp.eta()))/3.;

	     //Cheking that our best match has a good bx pattern

	     bufferBestAngularMatch.clear();
	     bufferBestAngularMatch.push_back(*Point);
	     bufferBestAngularMatch.push_back(*Point2);
	     bufferBestAngularMatch.push_back(*Point3);

	     std::sort(bufferBestAngularMatch.begin(), bufferBestAngularMatch.end(), bigmag);
	     
	     int lastbx=-7; 
	     bool increasing = false;
	     
	     for(std::vector<RPC4DHit>::iterator Pointbuff = bufferBestAngularMatch.begin(); Pointbuff!=bufferBestAngularMatch.end(); ++Pointbuff){
	       bool thisbx = false;           
	       if(lastbx<=Pointbuff->bx){
		 thisbx = true;
		 if(lastbx!= -7 && lastbx<Pointbuff->bx){
		   increasing=true;
		 } 
	       }
	       increasing = increasing*thisbx;
	       lastbx = Pointbuff->bx; 
	     }
	     //------------------------------------------------------
	     
	     if(increasing){
	       BestAngularMatch.clear();
	       BestAngularMatch.push_back(*Point);
	       BestAngularMatch.push_back(*Point2);
	       BestAngularMatch.push_back(*Point3);
	     }
	   }
	 }
       }}
     }}
   }

   bool hscp = false;
   foundvalue= false;
   bool increasingbx = false;
 
   if(minetaspread!=100.){
     std::cout<<"Inside the class "<<" candidate phi="<<phivalue<<" angularspread"<<minangularspread<<std::endl;
     std::cout<<"Inside the class "<<" candidate eta="<<etavalue<<" etaspread"<<minetaspread<<std::endl;
     
     std::sort(BestAngularMatch.begin(), BestAngularMatch.end(), bigmag);
     
     int lastbx=-7; 
     
     hscp = true;
     
     for(std::vector<RPC4DHit>::iterator Point = BestAngularMatch.begin(); Point!=BestAngularMatch.end(); ++Point){
       bool thisbx = false;           
       if(lastbx<=Point->bx){
	 thisbx = true;
	 if(lastbx!= -7 && lastbx<Point->bx){
	   increasingbx=true;
	 }
	 std::cout<<"Inside the class "<<Point->gp.mag()<<" "<<Point->bx<<" comparing "<<lastbx<<" with "<<Point->bx<<" Increasing bool="<<increasingbx<<std::endl;
       }
       hscp = hscp*thisbx;
       lastbx = Point->bx; 
       std::cout<<"Inside the class hscp="<<hscp;
     }
     std::cout<<"Inside the class "<<std::endl;
   }
   
   std::cout<<"Inside the class "<<" bool Increasing BX "<<increasingbx<<std::endl;
   
   hscp = hscp*increasingbx;
   
   //if(hscp) foundvalue = 1;
   //else foundvalue = 0;
   foundvalue = hscp;

   std::cout<<"Inside the class "<<" bool foundvalue = "<<foundvalue<<std::endl;
   

   //from now on beta estimation rpc standalone
   
   if(hscp){ 
     std::cout<<"Inside the class "<<" Candidate phi="<<phivalue<<" eta="<<etavalue<<std::endl;
     float beta =0;
     betavalue =0;

     bool anydifferentzero = true;
     bool anydifferentone = true;
     
     int lastbx=-7;
     bool increasing = true;

     for(std::vector<RPC4DHit>::iterator point = BestAngularMatch.begin(); point!=BestAngularMatch.end(); ++point){
       float r=point->gp.mag();
       increasing &= (point->bx>=lastbx); //condition 2: BX must be increase when going inside-out.
       anydifferentzero &= (!point->bx==0); //to check one knee withoutzeros
       anydifferentone &= (!point->bx==1); //to check one knee withoutones
       lastbx = point->bx;
       std::cout<<"Inside the class"<<"\t \t  r="<<r<<" phi="<<point->gp.phi()<<" eta="<<point->gp.eta()<<" bx="<<point->bx<<" increasing"<<increasing<<" anydifferentzero"<<anydifferentzero<<std::endl;
     }
     
     for(std::vector<RPC4DHit>::iterator point = BestAngularMatch.begin(); point!=BestAngularMatch.end(); ++point){
       std::cout<<point->bx;
      }
      std::cout<<std::endl;
      
     std::cout<<"Inside the class"<<"\t \t \t yes! We found an HSCPs let's try to stimate beta"<<std::endl;
     // here we should get some pattern-based estimate
     
     //Counting knees
     lastbx=-7;
     int knees=0;
     float maginknee = 0;
     float maginfirstknee = 0;
     
     for(std::vector<RPC4DHit>::iterator point = BestAngularMatch.begin(); point!=BestAngularMatch.end(); ++point){
       if(lastbx==-7){
	 maginfirstknee = point->gp.mag();
       }else if((lastbx!=point->bx)){
	 std::cout<<"Inside the class"<<"\t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
	 maginknee=point->gp.mag();
	 knees++;
       }
       lastbx=point->bx;
     }
      
     if(knees==0){
       std::cout<<"Inside the class"<<"\t \t \t \t knees="<<knees<<std::endl;
       beta=maginfirstknee/(25+maginfirstknee/30.)/30.;
     }else if(knees==1){
       float beta1=0;
       float beta2=0;
       std::cout<<"Inside the class"<<"\t \t \t \t knees="<<knees<<std::endl;
       std::cout<<"Inside the class"<<"\t \t \t \t anydifferentzero="<<anydifferentzero<<" anydifferentone="<<anydifferentone<<std::endl;
       if(!anydifferentzero){
	 beta=maginknee/(25+maginknee/30.)/30.;
       }else if(!anydifferentone){//i.e non zeros and no ones
	 beta=maginknee/(50+maginknee/30.)/30.;
       }else{
	 beta1=maginknee/(25+maginknee/30.)/30.;
	 float dr =(maginknee-maginfirstknee);
	 beta2 = dr/(25.+dr/30.);
	 std::cout<<"Inside the class"<<"\t \t \t \t \t not zero neither ones beta1="<<beta1<<" beta2="<<beta2<<std::endl;
	 beta = (beta1 + beta2)*0.5;
       }
     }else if(knees==2){
       std::cout<<"Inside the class"<<"\t \t \t \t knees="<<knees<<std::endl;
       knees=0;
       float beta1=0;
       float beta2=0;
       lastbx=-7;
       std::cout<<"Inside the class"<<"\t \t \t \t looping again on the RPCRecHits4D="<<knees<<std::endl;
       for(std::vector<RPC4DHit>::iterator point = BestAngularMatch.begin(); point!=BestAngularMatch.end(); ++point){
	 if(lastbx==-7){
	   maginfirstknee = point->gp.mag();
	 }else if((lastbx!=point->bx)){
	   std::cout<<"Inside the class"<<"\t \t \t \t \t one knee between"<<lastbx<<point->bx<<std::endl;
	   knees++;
	   if(knees==2){
	     float maginsecondknee=point->gp.mag();
	     beta1=maginknee/(25+maginknee/30.)/30.;
	     float dr =(maginknee-maginsecondknee);
	     beta2 = dr/(25.+dr/30.);
	     std::cout<<"Inside the class"<<"\t \t \t \t \t beta1="<<beta1<<" beta2="<<beta2<<std::endl;
	   }
	 }
	 lastbx=point->bx;
       }
       beta = (beta1 + beta2)*0.5;
     }
     
     betavalue = beta;
     
     std::cout<<"Inside the class \t \t \t beta="<<beta<<std::endl;
   }else std::cout<<"Inside the class "<<"No Candidate HSCP"<<std::endl;
   
   std::cout<<"Inside the class lastfoundvalue="<<foundvalue<<std::endl;
}

