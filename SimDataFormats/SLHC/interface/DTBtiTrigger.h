#ifndef __L1TriggerBTI__
#define __L1TriggerBTI__

#include <vector>

#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

using namespace std;

class DTBtiTrigger: public DTBtiTrigData
{
 public:
  DTBtiTrigger();
  DTBtiTrigger(const DTBtiTrigData& bti);
  DTBtiTrigger(const DTBtiTrigData& bti, 
	       Global3DPoint position,
	       Global3DVector direction);
  ~DTBtiTrigger() {}
  void setCMSPosition(const GlobalPoint pos)   { _position = pos; }
  void setCMSDirection(const GlobalVector dir) { _direction = dir; }
  Global3DPoint  cmsPosition()  const { return _position; }
  Global3DVector cmsDirection() const { return _direction; }
  std::string sprint() const;

 private:
  int _wheel;
  int _station;
  int _sector;
  int _superLayer;
  Global3DPoint  _position;
  Global3DVector _direction;
};

typedef vector<DTBtiTrigger> BtiTrigsCollection;

#endif

