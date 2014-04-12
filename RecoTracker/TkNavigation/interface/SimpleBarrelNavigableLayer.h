#ifndef TkNavigation_SimpleBarrelNavigableLayer_H
#define TkNavigation_SimpleBarrelNavigableLayer_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include <vector>


/** A concrete NavigableLayer for the barrel 
 */

class SimpleBarrelNavigableLayer GCC11_FINAL : public SimpleNavigableLayer {

public:

  SimpleBarrelNavigableLayer( BarrelDetLayer* detLayer,
			      const BDLC& outerBLC, 
			      const FDLC& outerLeftFL, 
			      const FDLC& outerRightFL,
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
  
  virtual void setInwardLinks(const BDLC& theBarrelv, const FDLC& theForwardv,TkLayerLess sorter = TkLayerLess(outsideIn));

private:
  BarrelDetLayer*   theDetLayer;
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
