#ifndef __L1TriggerChambPHI__
#define __L1TriggerChambPHI__

#include <vector>

#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"


using namespace std;


class DTTSPhiTrigger: public DTChambPhSegm 
{
  
 public:
  DTTSPhiTrigger();
  DTTSPhiTrigger(const DTChambPhSegm& c, 
		 Global3DPoint position,
		 Global3DVector direction);
  ~DTTSPhiTrigger() {}
  Global3DPoint cmsPosition()   const { return _position; }
  Global3DVector cmsDirection() const { return _direction; }
  std::string sprint() const;

 private:
  int _wheel;
  int _station;
  int _sector;
  int _psi;
  int _psiR;
  int _DeltaPsiR;
  float _phiB;
  Global3DPoint  _position;
  Global3DVector _direction;
};

 
typedef vector<DTTSPhiTrigger> TSPhiTrigsCollection;


#endif
