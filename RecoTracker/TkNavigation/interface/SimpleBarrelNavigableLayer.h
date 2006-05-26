#ifndef TkNavigation_SimpleBarrelNavigableLayer_H
#define TkNavigation_SimpleBarrelNavigableLayer_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include <vector>


/** A concrete NavigableLayer for the barrel 
 */

class SimpleBarrelNavigableLayer : public SimpleNavigableLayer {

public:

  SimpleBarrelNavigableLayer( BarrelDetLayer* detLayer,
			      const BDLC& outerBLC, 
			      const FDLC& outerLeftFL, 
			      const FDLC& outerRightFL,
			      const MagneticField* field,
			      float epsilon);

  SimpleBarrelNavigableLayer( BarrelDetLayer* detLayer,
			      const BDLC& outerBLC, 
			      const BDLC& innerBLC,
			      const BDLC& allOuterBLC,
			      const BDLC& allInnerBLC,
			      const FDLC& outerLeftFL, 
			      const FDLC& outerRightFL,
			      const FDLC& allOuterLeftFL,
			      const FDLC& allOuterRightFL,
			      const FDLC& innerLeftFL,
			      const FDLC& innerRightFL,
			      const FDLC& allInnerLeftFL,
			      const FDLC& allInnerRightFL,
			      const MagneticField* field,
			      float epsilon);
  
  // NavigableLayer interface
  virtual vector<const DetLayer*> 
  nextLayers( PropagationDirection timeDirection) const;

  virtual vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const;

  virtual vector<const DetLayer*> 
  compatibleLayers( PropagationDirection timeDirection) const;

  virtual vector<const DetLayer*> 
  compatibleLayers( const FreeTrajectoryState& fts, 
		    PropagationDirection timeDirection) const;


  // extended interface
  BDLC nextBarrelLayers()  { return theOuterBarrelLayers;}
  //  FDLC nextForwardLayers() { return theOuterForwardLayers;}
  
  virtual DetLayer* detLayer() const;
  virtual void   setDetLayer( DetLayer* dl);
  
  virtual void setInwardLinks(const BDLC& theBarrelv, const FDLC& theForwardv);

private:
  bool areAllReachableLayersSet;

  BarrelDetLayer*   theDetLayer;
  BDLC              theOuterBarrelLayers;
  BDLC              theInnerBarrelLayers;
  BDLC              theAllOuterBarrelLayers;
  BDLC              theAllInnerBarrelLayers;

  FDLC              theOuterLeftForwardLayers;
  FDLC              theOuterRightForwardLayers;
  FDLC              theAllOuterLeftForwardLayers;
  FDLC              theAllOuterRightForwardLayers;

  FDLC              theInnerLeftForwardLayers;
  FDLC              theInnerRightForwardLayers;
  FDLC              theAllInnerLeftForwardLayers;
  FDLC              theAllInnerRightForwardLayers;

  DLC               theNegOuterLayers;
  DLC               thePosOuterLayers;
  DLC               theNegInnerLayers;
  DLC               thePosInnerLayers;

  const BDLC& barrelLayers( const FreeTrajectoryState& fts,
			    PropagationDirection dir) const;

  const FDLC& forwardLayers( const FreeTrajectoryState& fts,
		       PropagationDirection dir) const;
};

#endif // SimpleBarrelNavigableLayer_H
