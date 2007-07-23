#include "RPCSim.h"


RPCSim::RPCSim(const edm::ParameterSet& config)
{
}


void
RPCSim::fillDigis(int rollDetId, RPCDigiCollection& digis)
{
  for (std::set< std::pair<int,int> >::iterator i=strips.begin();
       i!=strips.end(); i++){
    RPCDigi rpcDigi(i->first,i->second);
    //NCA
    digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
  }
  strips.clear();
}
