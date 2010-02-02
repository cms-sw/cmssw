#ifndef __L1TriggerTSTHETA__
#define __L1TriggerTSTHETA__

#include <vector>

#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"


/*
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

using namespace std;

class DTTSThetaTrigger: public DTChambThSegm
{
 public:
  DTTSThetaTrigger();
  DTTSThetaTrigger(const DTChambThSegm& tstheta);
  DTTSThetaTrigger(const DTChambThSegm& tstheta, 
		   Global3DPoint position,
		   Global3DVector direction);
  ~DTTSThetaTrigger() {}
  void setCMSPosition(const GlobalPoint pos)   { _position = pos; }
  void setCMSDirection(const GlobalVector dir) { _direction = dir; }
  Global3DPoint  cmsPosition()  const { return _position; }
  Global3DVector cmsDirection() const { return _direction; }
  std::string sprint() const;
  
 private:
  int _wheel;
  int _station;
  int _sector;
  Global3DPoint  _position;
  Global3DVector _direction;
};

typedef vector<DTTSThetaTrigger> TSThetaTrigsCollection;
*/

typedef vector<DTChambThSegm> TSThetaTrigsCollection;


#endif
