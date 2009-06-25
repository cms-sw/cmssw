#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiL1Link.h"

RPCDigiL1Link::RPCDigiL1Link()
{
  for (unsigned int i=0;i<7;i++){
    std::pair<unsigned int, int > c(0,0);
    _link.push_back(c);
  }
}

    
RPCDigiL1Link::~RPCDigiL1Link()
{}


bool 
RPCDigiL1Link::empty() const
{
  bool e=true;
  for (unsigned int l=1;l<=6;l++){
    if (!this->empty(l)) e=false;
  } 
  return e;
}


bool 
RPCDigiL1Link::empty(unsigned int layer) const
{
  return this->rawdetId(layer)==0;
}


unsigned int 
RPCDigiL1Link::rawdetId(unsigned int layer) const
{
  return _link[layer-1].first;
}
 

int 
RPCDigiL1Link::strip(unsigned int layer) const
{
  return abs(_link[layer-1].second)%1000;
}

int 
RPCDigiL1Link::bx(unsigned int layer) const
{
  return _link[layer-1].second/1000;
}

unsigned int
RPCDigiL1Link::nlayer() const
{
  return _link.size()-1;
}


void 
RPCDigiL1Link::setLink(unsigned int layer, 
		       unsigned int rpcdetId, 
		       int strip, int bx)
{

  int pdigi=abs(bx)*1000+strip;
  if (bx<0)
    pdigi*=-1;
  std::pair<unsigned int, int> digi(rpcdetId,pdigi);
  _link[layer-1]=digi;

}
