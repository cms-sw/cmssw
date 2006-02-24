#ifndef Vertex_BeamSpot_H
#define Vertex_BeamSpot_H

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"

/** \class BeamSpot 
 *  Defines position and spread of the beam spot 
 *  by x, y, z coordinates and sigma_x, sigma_y, sigma_z spreads. 
 *  All quantities are configurable in .orcarc
 */

class BeamSpot {
public:

  /** Configurable constructor
   *  Default position is (0, 0, 0)
   *  Default spread is (15 mum, 15 mum, 5.3 cm), 
   *  i.e. error matrix is diagonal with elements
   *  (0.0015*0.0015, 0.0015*0.0015, 5.3*5.3)
   */
  BeamSpot();
  
  GlobalPoint position() const { return thePos; }
  GlobalError error() const { return theErr; }

private:

  GlobalPoint thePos;
  GlobalError theErr;

};
#endif  //  Vertex_BeamSpot_H
