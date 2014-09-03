#ifndef TkNavigation_SimpleForwardNavigableLayer_H
#define TkNavigation_SimpleForwardNavigableLayer_H

#include "SimpleNavigableLayer.h"

/** A concrete NavigableLayer for the forward
 */

class dso_hidden SimpleForwardNavigableLayer GCC11_FINAL : public SimpleNavigableLayer {

public:

  SimpleForwardNavigableLayer( const ForwardDetLayer* detLayer,
			       const BDLC& outerBL, 
			       const FDLC& outerFL, 
			       const MagneticField* field,
			       float epsilon,
			       bool checkCrossingSide=true);

  // NavigableLayer interface
  virtual std::vector<const DetLayer*> 
  nextLayers( NavigationDirection direction) const override;

  virtual std::vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const override;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( NavigationDirection direction) const override;

  virtual std::vector<const DetLayer*> 
  compatibleLayers( const FreeTrajectoryState& fts, 
		    PropagationDirection dir) const override {
    int counter=0;
    return SimpleNavigableLayer::compatibleLayers(fts,dir,counter);
  }

  virtual void setAdditionalLink(const DetLayer*, NavigationDirection direction=insideOut) override;

  virtual const DetLayer* detLayer() const override { return theDetLayer;} 
  virtual void   setDetLayer( const DetLayer* dl) override;

  virtual void setInwardLinks( const BDLC&, const FDLC&, TkLayerLess sorter = TkLayerLess(outsideIn)) override;

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



