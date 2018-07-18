#ifndef TkNavigation_SimpleForwardNavigableLayer_H
#define TkNavigation_SimpleForwardNavigableLayer_H

#include "SimpleNavigableLayer.h"

/** A concrete NavigableLayer for the forward
 */

class dso_hidden SimpleForwardNavigableLayer final : public SimpleNavigableLayer {

public:

  SimpleForwardNavigableLayer( const ForwardDetLayer* detLayer,
			       const BDLC& outerBL, 
			       const FDLC& outerFL, 
			       const MagneticField* field,
			       float epsilon,
			       bool checkCrossingSide=true);

  // NavigableLayer interface
  std::vector<const DetLayer*> 
  nextLayers( NavigationDirection direction) const override;

  std::vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const override;

  using SimpleNavigableLayer::compatibleLayers;

  std::vector<const DetLayer*> 
  compatibleLayers( NavigationDirection direction) const override;

  std::vector<const DetLayer*> 
  compatibleLayers( const FreeTrajectoryState& fts, 
		    PropagationDirection dir) const override {
    int counter=0;
    return SimpleNavigableLayer::compatibleLayers(fts,dir,counter);
  }

  void setAdditionalLink(const DetLayer*, NavigationDirection direction=insideOut) override;

  const DetLayer* detLayer() const override { return theDetLayer;} 
  void   setDetLayer( const DetLayer* dl) override;

  void setInwardLinks( const BDLC&, const FDLC&, TkLayerLess sorter = TkLayerLess(outsideIn)) override;

private:
  const ForwardDetLayer*  theDetLayer;
  BDLC              theOuterBarrelLayers;
  BDLC              theInnerBarrelLayers;

  FDLC              theOuterForwardLayers;
  FDLC              theInnerForwardLayers;

  DLC               theOuterLayers;
  DLC               theInnerLayers;

};

#endif // SimpleForwardNavigableLayer_H



