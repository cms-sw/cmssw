#ifndef TR_FastCircle_H_
#define TR_FastCircle_H_
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

/**
   Calculate circle parameters (x0, y0, rho) for a circle:
   (x-x0)^2 + (y-y0)^2 = rho^2 
   in Global Cartesian Coordinates in the (x,y) plane for a given set of 
   GlobalPoints. It is done by mapping the points onto the Riemann Sphere 
   and fit a plane to the transformed coordinates of the points. 
   The method is described in:

   A.Strandlie, J.Wroldsen, R.Fruehwirth, B.Lillekjendlie:
   Particle tracks fitted on the Riemann sphere
   Computer Physics Communications 131 (2000) 95-108, 18 January 2000
   
   Implementation: Matthias Winkler, 14 February 2001

   This implementation is a specialized version of the general Circle class
   for three points.

   Update 14.02.2001: For 3 Points (2 RecHits + Vertex) the plain parameters
                      n1*x + n2*y + n3*z + c = 0
                      are analytically calculable.
   Update 14.02.2001: In the case that a circle fit is not possible (points 
                      are along a straight line) the parameters of the
		      straight line can be used:
		      c + n1*x + n2*y = 0 
 */

class FastCircle {

public:
  
  FastCircle(const GlobalPoint& outerHit,
	     const GlobalPoint& middleHit,
	     const GlobalPoint& aVertex);

  FastCircle(const GlobalPoint& outerHit,
	     const GlobalPoint& middleHit,
	     const GlobalPoint& aVertex,
	     double norm);
  
  ~FastCircle() {}
  
  // all returned values have dimensions of cm
  // parameters of the circle (circle is valid)
  double x0() const {return theX0;}
  
  double y0() const {return theY0;}

  double rho() const {return theRho;}
  
  bool isValid() const {return theValid;}
  
  // parameters of the straight line 
  // (if circle is invalid only these are available) 
  double n1() const {return theN1;}
  
  double n2() const {return theN2;}
  
  double c() const {return theC;}
  
  GlobalPoint const & outerPoint() const { return theOuterPoint;} 
  GlobalPoint const & innerPoint() const { return theInnerPoint;} 
  GlobalPoint const & vertexPoint() const { return theVertexPoint;} 


private:

  GlobalPoint theOuterPoint; 
  GlobalPoint theInnerPoint; 
  GlobalPoint theVertexPoint; 
  double theNorm;
  
  double theX0;
  double theY0;
  double theRho;
  
  double theN1;
  double theN2;
  double theC;
  
  bool theValid;
  
  void createCircleParameters() dso_hidden;
   
};

#endif //TR_Circle_H_
