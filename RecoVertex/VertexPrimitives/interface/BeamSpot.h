#ifndef Vertex_BeamSpot_H
#define Vertex_BeamSpot_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** \class BeamSpot 
 * Beamspot class to be used in vertex reconstruction. 
 * Constructed from either a reco::BeamSpot or a BeamSpotObjects.
 * It takes the tilt of the beamline into account by rotating the original diagonal 
 * covariance matrix.
 */

class BeamSpot {
public:

  /** 
   *  Default constructor. Default position is (0, 0, 0). No tilt of the beamline.
   *  Default spread is (15 mum, 15 mum, 5.3 cm), 
   *  i.e. error matrix is diagonal with elements
   *  (0.0015*0.0015, 0.0015*0.0015, 5.3*5.3)
   */
  BeamSpot();
  BeamSpot(const reco::BeamSpot & beamSpot);
  BeamSpot(const BeamSpotObjects * beamSpot);

  GlobalPoint position() const { return thePos; }
  GlobalError error() const { return theErr; }

private:

  AlgebraicSymMatrix rotateMatrix(const GlobalVector& newZ,
	const AlgebraicSymMatrix& diagError) const;

  GlobalPoint thePos;
  GlobalError theErr;

};
#endif  //  Vertex_BeamSpot_H
