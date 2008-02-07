#ifndef RPCOBJECTS_RPCDIGISIMLINK_H
#define RPCOBJECTS_RPCDIGISIMLINK_H

#include "boost/cstdint.hpp"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <map>

class RPCDigiSimLink {
 public:
  RPCDigiSimLink(std::pair<unsigned int,int> digi, const PSimHit* simhit):_simhit(simhit)
    {
      _digi = digi;
     
    }

  RPCDigiSimLink():_digi(0,0),_simhit(0)
    {;}

  ~RPCDigiSimLink(){;}

  unsigned int  getStrip()     const {return _digi.first;}
  unsigned int  getBx()     const {return _digi.second;}
  const PSimHit* getSimHit() const {return _simhit;}

  inline bool operator< ( const RPCDigiSimLink& other ) const { return getStrip() < other.getStrip(); }

 private:
  std::pair<unsigned int,int> _digi;
  const PSimHit* _simhit;

};
#endif


  
