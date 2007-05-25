#ifndef TR_FastLine_H_
#define TR_FastLine_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

/**
   Calculate the Line parameters (n1, n2, c) of a Line in Rho*Phi-Z. A Line
   is defined by 
   n1*x + n2*y + c = 0. (== n1*RHOPHI + n2*Z + c)
   If rho is not specified, the Line parameters are calculated in R-Z.
   
   Implementation: Matthias Winkler 21.02.2001
 */

class FastLine {

public:
  
  FastLine(const GlobalPoint& outerHit,
	   const GlobalPoint& innerHit);

  FastLine(const GlobalPoint& outerHit,
	   const GlobalPoint& innerHit,
	   double rho);
  
  ~FastLine() {}
  
  double n1() const {return theN1;}
  
  double n2() const {return theN2;}
  
  double c() const {return theC;}

  bool isValid() const {return theValid;}
    
private:

  GlobalPoint theOuterHit; 
  GlobalPoint theInnerHit; 
  double theRho;
  
  double theN1;
  double theN2;
  double theC;
  
  bool theValid;
  
  void createLineParameters();
  
};

#endif //TR_FastLine_H_
