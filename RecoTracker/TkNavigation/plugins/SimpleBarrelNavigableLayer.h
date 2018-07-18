#ifndef TkNavigation_SimpleBarrelNavigableLayer_H
#define TkNavigation_SimpleBarrelNavigableLayer_H

#include "SimpleNavigableLayer.h"

#include <vector>


/** A concrete NavigableLayer for the barrel 
 */

class dso_hidden SimpleBarrelNavigableLayer final : public SimpleNavigableLayer {

public:

  SimpleBarrelNavigableLayer( BarrelDetLayer const* detLayer,
			      const BDLC& outerBLC, 
			      const FDLC& outerLeftFL, 
			      const FDLC& outerRightFL,
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
  
  void setInwardLinks(const BDLC& theBarrelv, const FDLC& theForwardv,TkLayerLess sorter = TkLayerLess(outsideIn)) override;

private:
  const BarrelDetLayer*   theDetLayer;
  BDLC              theOuterBarrelLayers;
  BDLC              theInnerBarrelLayers;

  FDLC              theOuterLeftForwardLayers;
  FDLC              theOuterRightForwardLayers;

  FDLC              theInnerLeftForwardLayers;
  FDLC              theInnerRightForwardLayers;

  DLC               theNegOuterLayers;
  DLC               thePosOuterLayers;
  DLC               theNegInnerLayers;
  DLC               thePosInnerLayers;

};

#endif // SimpleBarrelNavigableLayer_H
