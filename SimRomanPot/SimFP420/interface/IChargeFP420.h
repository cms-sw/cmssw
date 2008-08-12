#ifndef IChargeFP420_h
#define IChargeFP420_h

#include "SimRomanPot/SimFP420/interface/CDrifterFP420.h"

#include<map>

// induce signal on electrods
class IChargeFP420{
 public:
  
  typedef map< int, float, less<int> > hit_map_type;
  
  
  virtual ~IChargeFP420() { }
  virtual hit_map_type induce(CDrifterFP420::collection_type, int, double, int, double, int) = 0 ;
};
#endif
