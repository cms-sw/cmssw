#ifndef _TRACKER_EnergyDepositUnit_H
#define _TRACKER_EnergyDepositUnit_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
/**
 * Class which allows to "follow" an elementary charge in the silicon.
 * It basically defines a quantum of energy in the bulk, with a 3D position.
 */
class EnergyDepositUnit{
 public:
  EnergyDepositUnit(): _energy(0),_position(0,0,0){}
  EnergyDepositUnit(float energy,float x, float y, float z):
    _energy(energy),_position(x,y,z){}
  EnergyDepositUnit(float energy, const Local3DPoint& position):
    _energy(energy),_position(position){}
  float x() const{return _position.x();}
  float y() const{return _position.y();}
  float z() const{return _position.z();}
  float energy() const { return _energy;}
 private:
  float _energy;
  Local3DPoint _position;  
};


#endif
