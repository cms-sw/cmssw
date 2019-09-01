#ifndef EnergySegmentFP420_h
#define EnergySegmentFP420_h

#include <vector>

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "G4StepPoint.hh"

// define a quantum of energy and position.
class EnergySegmentFP420 {
public:
  EnergySegmentFP420() : _energy(0), _position(0, 0, 0) {}

  EnergySegmentFP420(float energy, float x, float y, float z) : _energy(energy), _position(x, y, z) {}

  //      EnergySegmentFP420(float energy, G4ThreeVector position):
  //	_energy(energy),_position(position){}

  EnergySegmentFP420(float energy, Local3DPoint position) : _energy(energy), _position(position) {}

  float x() const { return _position.x(); }
  float y() const { return _position.y(); }
  float z() const { return _position.z(); }
  float energy() const { return _energy; }

private:
  float _energy;
  //	G4ThreeVector  _position;
  Local3DPoint _position;
};

#endif
