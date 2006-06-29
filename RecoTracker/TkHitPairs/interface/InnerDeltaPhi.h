#ifndef InnerDeltaPhi_H
#define InnerDeltaPhi_H

/** predict phi bending in layer for the tracks constratind by outer hit r-z */ 
#include <fstream>
#include "FWCore/Framework/interface/EventSetup.h"
//#include "Utilities/Notification/interface/TimingReport.h"

class DetLayer;
class MultipleScatteringParametrisation;

class InnerDeltaPhi {
public:

  InnerDeltaPhi( const DetLayer& layer, 
		 float ptMin,  float rOrigin,
		 float zMinOrigin, float zMaxOrigin,const edm::EventSetup& iSetup,
		 bool precise = true);

   ~InnerDeltaPhi();

  float operator()( float rHit, float zHit, float errRPhi = 0.) const;

private:

  float theROrigin;
  float theRLayer;
  float theRCurvature;
  float theHitError;
  float theA;
  float theB;
  bool  theRDefined;
  float theVtxZ;
  float thePtMin;
  MultipleScatteringParametrisation * sigma;
  bool thePrecise;

  void initBarrelLayer( const DetLayer& layer);
  void initForwardLayer( const DetLayer& layer, 
			 float zMinOrigin, float zMaxOrigin);

  float minRadius( float hitR, float hitZ) const {
    if (theRDefined) return theRLayer;
    else {
      float invRmin = (hitZ-theB)/theA/hitR;
      return ( invRmin> 0) ? std::max( 1./invRmin, (double)theRLayer) : theRLayer;
    }
  }

  // the timers are static because we want the same timer for all instances
 /*  static TimingReport::Item * theConstructTimer; */
/*   static TimingReport::Item * theDeltaPhiTimer; */
/*   static bool theTimingDone; */
/*   void initTiming(); */

};

#endif
