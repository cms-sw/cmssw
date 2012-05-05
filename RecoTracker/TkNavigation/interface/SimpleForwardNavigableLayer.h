#ifndef TkNavigation_SimpleForwardNavigableLayer_H
#define TkNavigation_SimpleForwardNavigableLayer_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

/** A concrete NavigableLayer for the forward
 */

class SimpleForwardNavigableLayer GCC11_FINAL : public SimpleNavigableLayer {

public:

  SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			       const BDLC& outerBL, 
			       const FDLC& outerFL, 
			       const MagneticField* field,
			       float epsilon,
			       bool checkCrossingSide=true);

  SimpleForwardNavigableLayer( ForwardDetLayer* detLayer,
			       const BDLC& outerBL, 
			       const BDLC& allOuterBL,
			       const BDLC& innerBL,
			       const BDLC& allInnerBL,
			       const FDLC& outerFL, 
			       const FDLC& allOuterFL,
			       const FDLC& innerFL,
			       const FDLC& allInnerFL,
			       const MagneticField* field,
			       float epsilon,
			       bool checkCrossingSide=true);

  // NavigableLayer interface
  virtual std::vector<const DetLayer*> 
  nextLayers( NavigationDirection direction) const;

  virtual std::vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( NavigationDirection direction) const;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( const FreeTrajectoryState& fts, 
		    PropagationDirection timeDirection) const;

  virtual void setAdditionalLink(DetLayer*, NavigationDirection direction=insideOut);

  virtual DetLayer* detLayer() const;
  virtual void   setDetLayer( DetLayer* dl);

  virtual void setInwardLinks( const BDLC&, const FDLC&, TkLayerLess sorter = TkLayerLess(outsideIn));

private:
  bool areAllReachableLayersSet;

  ForwardDetLayer*  theDetLayer;
  BDLC              theOuterBarrelLayers;
  BDLC              theAllOuterBarrelLayers;
  BDLC              theInnerBarrelLayers;
  BDLC              theAllInnerBarrelLayers;
  FDLC              theOuterForwardLayers;
  FDLC              theAllOuterForwardLayers;
  FDLC              theInnerForwardLayers;
  FDLC              theAllInnerForwardLayers;
  DLC               theOuterLayers;
  DLC               theInnerLayers;
  DLC               theAllOuterLayers;
  DLC               theAllInnerLayers;

};

#endif // SimpleForwardNavigableLayer_H



