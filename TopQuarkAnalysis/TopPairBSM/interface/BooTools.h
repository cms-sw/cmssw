#ifndef BooTools_h
#define BooTools_h

/**_________________________________________________________________
   class:   BooTools.h
   package: TopPairBSM


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BooTools.h,v 1.1.2.1 2009/03/08 03:26:21 yumiceva Exp $

________________________________________________________________**/


#include "TLorentzVector.h"

class BooTools {

  public:
	
	BooTools();
	~BooTools();

	/// wiggle jets to match a given mass
	double fix4VectorsForMass( TLorentzVector &vec1, TLorentzVector &vec2,
							   double targetMass,
							   double upperwidth1, double upperwidth2,
							   double lowerwidth1=-1, double lowerwidth2=-1);
	
  private:
	


};

#endif

