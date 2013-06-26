#include "RPCFakeEvent.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"



#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


RPCFakeEvent::RPCFakeEvent(const edm::ParameterSet& config) {

  std::cout <<"Initialize the Event Dump"<<std::endl;
  produces<RPCDigiCollection>();


  filesed=config.getParameter<std::vector<std::string> >("FakeEvents");
  rpcdigiprint = config.getParameter<bool>("printOut"); 
  std::cout<<"Number of Input files ="<<filesed.size()<<std::endl;
  if (rpcdigiprint) {
    std::cout<<"Event Dump Digi Creation"<<std::endl;
    std::cout<<"Number of Input files ="<<filesed.size()<<std::endl;
    for (std::vector<std::string>::iterator i=filesed.begin();
	 i<filesed.end(); i++){
      std::cout <<"input file "<<*i<<std::endl;
    }
  }
}

void
RPCFakeEvent::produce(edm::Event & e, const edm::EventSetup& c)
{

  std::cout <<"Getting the rpc geometry"<<std::endl;
  edm::ESHandle<RPCGeometry> rpcGeom;
  std::cout <<"Getting the Muon geometry"<<std::endl;
  c.get<MuonGeometryRecord>().get(rpcGeom);

  if (rpcdigiprint){

    std::cout <<" Evento Done : "
	      <<"run="<<e.id().run()
	      <<" event="<<e.id().event()<<std::endl;
  }
  if ( e.id().event() <= filesed.size() ) {
    if (rpcdigiprint){
      std::cout<<"Opening file "<<filesed[e.id().event()-1]<<std::endl;
    }
  }
  std::auto_ptr<RPCDigiCollection> pDigis(new RPCDigiCollection());

  {
    std::cout <<"Station 1 RB1in"<<std::endl;
    RPCDetId r(0,1,1,10,1,1,3);
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    //    RPCDigi rpcDigi1(67,0);
    //    RPCDigi rpcDigi2(68,0);
    //    RPCDigi rpcDigi3(69,0);
    RPCDigi rpcDigi1(22,0);
    RPCDigi rpcDigi2(23,0);
    RPCDigi rpcDigi3(24,0);
    pDigis->insertDigi(r,rpcDigi1);  
    pDigis->insertDigi(r,rpcDigi2);  
    pDigis->insertDigi(r,rpcDigi3);  
  }
  {
    std::cout <<"Station 1 RB1out"<<std::endl;
    RPCDetId r(0,1,1,10,2,1,3);
    std::cout <<" RB1 out "<<r<<std::endl;
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    //    RPCDigi rpcDigi1(63,0);
    // RPCDigi rpcDigi2(64,0);
    //RPCDigi rpcDigi3(65,0);

    RPCDigi rpcDigi1(20,0);
    RPCDigi rpcDigi2(21,0);
    RPCDigi rpcDigi3(22,0);
    pDigis->insertDigi(r,rpcDigi1);  
    pDigis->insertDigi(r,rpcDigi2);  
    pDigis->insertDigi(r,rpcDigi3);  
  }
  {
    RPCDetId r(0,1,2,10,1,1,3);
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    RPCDigi rpcDigi1(8,0);
    RPCDigi rpcDigi2(9,0);
    RPCDigi rpcDigi3(12,0);
    pDigis->insertDigi(r,rpcDigi1);  
    pDigis->insertDigi(r,rpcDigi2);  
    pDigis->insertDigi(r,rpcDigi3);  
  }
  {
    RPCDetId r(0,1,2,10,2,1,3);
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    RPCDigi rpcDigi1(8,0);
    RPCDigi rpcDigi2(9,0);
    pDigis->insertDigi(r,rpcDigi1);  
    pDigis->insertDigi(r,rpcDigi2);  
  }
  {
    RPCDetId r(0,1,3,10,1,2,3);
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    //    RPCDigi rpcDigi1(16,0);
    // RPCDigi rpcDigi2(17,0);
    RPCDigi rpcDigi1(23,0);
    RPCDigi rpcDigi2(24,0);

    pDigis->insertDigi(r,rpcDigi1);  
    pDigis->insertDigi(r,rpcDigi2);  
  }
  {
    RPCDetId r(0,1,4,10,1,2,3);
    const RPCRoll* roll = rpcGeom->roll(r);
    std::cout <<r<<" Number of strip "<<roll->nstrips()<<std::endl;
    RPCDigi rpcDigi1(46,0);
    pDigis->insertDigi(r,rpcDigi1);  
  }
  e.put(pDigis);
}

DEFINE_FWK_MODULE(RPCFakeEvent);

  
