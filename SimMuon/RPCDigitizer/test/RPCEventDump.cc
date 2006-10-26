#include "RPCEventDump.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"



#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


RPCEventDump::RPCEventDump(const edm::ParameterSet& config) {
  std::cout <<"Initialize the Event Dump"<<std::endl;
  produces<RPCDigiCollection>();

  filesed=config.getParameter<std::vector<std::string> >("EventDumps");
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
RPCEventDump::produce(edm::Event & e, const edm::EventSetup& c)
{
  if (rpcdigiprint){
    std::cout <<" Evento Done : "
	      <<"   -- run="<<e.id().run()
	      <<" event="<<e.id().event()<<std::endl;
  }
  if ( e.id().event() <= filesed.size() ) {
    if (rpcdigiprint){
      std::cout<<"Opening file "<<filesed[e.id().event()-1]<<std::endl;
    }
  }
  std::auto_ptr<RPCDigiCollection> pDigis(new RPCDigiCollection());

  {
    RPCDetId r(0,1,1,10,1,1,1);
    RPCDigi rpcDigi(1,0);
    pDigis->insertDigi(r,rpcDigi);  
  }
  {
    RPCDetId r(0,1,1,10,2,1,1);
    RPCDigi rpcDigi1(1,0);
    pDigis->insertDigi(r,rpcDigi1);  
    RPCDigi rpcDigi2(2,0);
    pDigis->insertDigi(r,rpcDigi2);  
  }
  {
    RPCDetId r(0,1,2,10,1,1,1);
    RPCDigi rpcDigi1(1,0);
    pDigis->insertDigi(r,rpcDigi1);  
    RPCDigi rpcDigi2(2,0);
    pDigis->insertDigi(r,rpcDigi2);  
    RPCDigi rpcDigi3(3,0);
    pDigis->insertDigi(r,rpcDigi3);  
  }
  {
    RPCDetId r(0,1,2,10,2,1,1);
    RPCDigi rpcDigi1(1,0);
    pDigis->insertDigi(r,rpcDigi1);  
    RPCDigi rpcDigi2(2,0);
    pDigis->insertDigi(r,rpcDigi2);  
    RPCDigi rpcDigi3(3,0);
    pDigis->insertDigi(r,rpcDigi3);  
    RPCDigi rpcDigi4(4,0);
    pDigis->insertDigi(r,rpcDigi4);  
  }
  {
    RPCDetId r(0,1,3,10,1,1,1);
    RPCDigi rpcDigi1(1,0);
    pDigis->insertDigi(r,rpcDigi1);  
    RPCDigi rpcDigi2(2,0);
    pDigis->insertDigi(r,rpcDigi2);  
    RPCDigi rpcDigi3(3,0);
    pDigis->insertDigi(r,rpcDigi3);  
    RPCDigi rpcDigi4(4,0);
    pDigis->insertDigi(r,rpcDigi4);  
    RPCDigi rpcDigi5(5,0);
    pDigis->insertDigi(r,rpcDigi5);  
  }
  {
    RPCDetId r(0,1,4,10,1,1,1);
    RPCDigi rpcDigi4(4,0);
    pDigis->insertDigi(r,rpcDigi4);  
    RPCDigi rpcDigi5(5,0);
    pDigis->insertDigi(r,rpcDigi5);  
    RPCDigi rpcDigi6(6,0);
    pDigis->insertDigi(r,rpcDigi6);  
  }
  e.put(pDigis);
}

DEFINE_FWK_MODULE(RPCEventDump);

  
