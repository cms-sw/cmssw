#ifndef TkNavigation_SimpleForwardNavigableLayer_H
#define TkNavigation_SimpleForwardNavigableLayer_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

/** A concrete NavigableLayer for the forward
 */

class SimpleForwardNavigableLayer : public SimpleNavigableLayer {

public:

  SimpleForwardNavigableLayer( ForwardDetLayer*,
			       const BDLC&, const FDLC&,const MagneticField* field, float);

  // NavigableLayer interface
  virtual vector<const DetLayer*> 
  nextLayers( PropagationDirection timeDirection) const;

  virtual vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const;

  virtual DetLayer* detLayer() const;
  virtual void   setDetLayer( DetLayer* dl);

  virtual void setInwardLinks( const BDLC&, const FDLC&);

private:

  ForwardDetLayer*  theDetLayer;
  BDLC              theOuterBarrelLayers;
  BDLC              theInnerBarrelLayers;
  FDLC              theOuterForwardLayers;
  FDLC              theInnerForwardLayers;
  DLC               theOuterLayers;
  DLC               theInnerLayers;

};

#endif // SimpleForwardNavigableLayer_H



