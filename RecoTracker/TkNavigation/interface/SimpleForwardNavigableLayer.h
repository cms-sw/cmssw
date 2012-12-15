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
		    PropagationDirection dir) const {
    int counter=0;
    return SimpleNavigableLayer::compatibleLayers(fts,dir,counter);
  }

  virtual void setAdditionalLink(DetLayer*, NavigationDirection direction=insideOut);

  virtual DetLayer* detLayer() const { return theDetLayer;}
  virtual void   setDetLayer( DetLayer* dl);

  virtual void setInwardLinks( const BDLC&, const FDLC&, TkLayerLess sorter = TkLayerLess(outsideIn));

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



