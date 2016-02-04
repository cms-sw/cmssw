#ifndef FP420HitsObject_H
#define FP420HitsObject_H

#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

class FP420HitsObject{
 public:
  FP420HitsObject(std::string, TrackingSlaveSD::Collection &);
  std::string name(){return _name;}
  TrackingSlaveSD::Collection& hits(){return _hits;}
 private:
  TrackingSlaveSD::Collection& _hits;
  std::string _name;

};


#endif
