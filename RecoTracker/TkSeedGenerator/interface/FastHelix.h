#ifndef TR_FastHelix_H_
#define TR_FastHelix_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

/**
   Generation of track parameters at a vertex using two hits and a vertex.
   It is used e.g. by a seed generator.

   24.01.2012: introduced Maxpt cut. changed algo of "FastLine" to use vertex
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

class MagneticField;
class FastHelix {
public:

  //Original constructor (no basis vertex)
  FastHelix(const GlobalPoint& oHit,
	    const GlobalPoint& mHit,
	    const GlobalPoint& aVertex,
	    double nomField, MagneticField const * ibField) :
    bField(ibField),
    theCircle(oHit,
	      mHit,
	      aVertex) {
    tesla0=0.1*nomField;
    maxRho = maxPt/(0.01 * 0.3*tesla0);
    useBasisVertex = false;
    compute();
  }

  //New constructor (with basis vertex)
  FastHelix(const GlobalPoint& oHit,
	    const GlobalPoint& mHit,
	    const GlobalPoint& aVertex,
	    double nomField, MagneticField const * ibField,
	    const GlobalPoint& bVertex) : 
    bField(ibField),
    basisVertex(bVertex),
    theCircle(oHit,
	      mHit,
	      aVertex) {
    tesla0=0.1*nomField;
    maxRho = maxPt/(0.01 * 0.3*tesla0);
    useBasisVertex = true;
    compute();
  }

  ~FastHelix() {}
  
  bool isValid() const {return theCircle.isValid();}

  GlobalTrajectoryParameters stateAtVertex() const { return atVertex; }

  const FastCircle & circle() const { return theCircle; }

private:

  GlobalPoint const & outerHit() const { return theCircle.outerPoint();} 
  GlobalPoint const & middleHit() const { return theCircle.innerPoint();} 
  GlobalPoint const & vertex() const { return theCircle.vertexPoint();} 


  void compute();
  void helixStateAtVertex() dso_hidden;
  void straightLineStateAtVertex() dso_hidden;


private:

  static constexpr float maxPt = 10000; // 10Tev

  MagneticField const * bField; // needed to construct GlobalTrajectoryParameters
  GlobalTrajectoryParameters atVertex;
  GlobalPoint basisVertex;
  FastCircle theCircle;
  float tesla0;
  float maxRho;
  bool useBasisVertex;
};

#endif //TR_FastHelix_H_

