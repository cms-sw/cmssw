#ifndef IChargeFP420_h
#define IChargeFP420_h

#include "SimRomanPot/SimFP420/interface/CDrifterFP420.h"

#include <map>

// induce signal on electrods
class IChargeFP420 {
public:
  typedef std::map<int, float, std::less<int>> hit_map_type;

  virtual ~IChargeFP420() {}
  virtual hit_map_type induce(const CDrifterFP420::collection_type &, int, double, int, double, int, int) = 0;
};
#endif
