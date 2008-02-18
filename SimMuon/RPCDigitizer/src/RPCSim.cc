#include "RPCSim.h"


RPCSim::RPCSim(const edm::ParameterSet& config)
{
}


void
RPCSim::fillDigis(int rollDetId, RPCDigiCollection& digis)
{

  for (std::set< std::pair<int,int> >::iterator i=strips.begin();
       i!=strips.end(); i++){
    if(i->second != -999){
      RPCDigi rpcDigi(i->first,i->second);
      //NCA
      digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
      //    std::cout<<"DIGI:"<<rpcDigi<<std::endl;
    }
  }
  strips.clear();
}
