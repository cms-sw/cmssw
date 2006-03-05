#ifndef TR_FastHelix_H_
#define TR_FastHelix_H_

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
// You don't really want to know why the rest does not compile on Sun 
// whthout this method, and why it compiles with it. 
inline GlobalPoint sun_bullshit( const GlobalPoint point) {
  return point;
}

/**
   Generation of track parameters at a vertex using two hits and a vertex.
   It is used e.g. by a seed generator.

   21.02.2001: Old FastHelix is now called FastHelixFit. Replace FastLineFit
               by FastLine (z0, dz/drphi calculated without vertex and errors)
   14.02.2001: Replace general Circle by FastCircle.
   13.02.2001: LinearFitErrorsInTwoCoordinates replaced by FastLineFit
   29.11.2000: (Pascal Vanlaer) Modification of calculation of sign of px,py
               and change in calculation of pz, z0.
   29.11.2000: (Matthias Winkler) Split stateAtVertex() in two parts (Circle 
               is valid or not): helixStateAtVertex() and 
	                         straightLineStateAtVertex()
 */

class FastHelix {

private:
  
  typedef FreeTrajectoryState FTS;

public:

  FastHelix(const GlobalPoint& outerHit, 
	    const GlobalPoint& middleHit,
	    const GlobalPoint& aVertex,
	    const edm::EventSetup& iSetup);
  
  ~FastHelix() {}
  
  bool isValid() const {return theCircle.isValid();}

  FTS stateAtVertex() const;

  FTS helixStateAtVertex() const;

  FTS straightLineStateAtVertex() const;

private:
  
  GlobalPoint theOuterHit;
  GlobalPoint theMiddleHit;
  GlobalPoint theVertex;
  FastCircle theCircle;
  edm::ESHandle<MagneticField> pSetup;
  GlobalVector tesla0;
};

#endif //TR_FastHelix_H_




