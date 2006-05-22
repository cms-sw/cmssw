#include "RPCSim.h"


RPCSim::RPCSim(const edm::ParameterSet& config)
{
}


void
RPCSim::fillDigis(int rollDetId, RPCDigiCollection& digis)
{
  for (std::set<int>::iterator i=strips.begin();
       i!=strips.end(); i++){
    RPCDigi rpcDigi(*i,0);
    //NCA
    digis.insertDigi(RPCDetId(rollDetId),rpcDigi);
  }
  strips.clear();
}
