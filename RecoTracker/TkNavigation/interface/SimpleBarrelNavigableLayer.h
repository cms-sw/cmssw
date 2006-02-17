#ifndef TkNavigation_SimpleBarrelNavigableLayer_H
#define TkNavigation_SimpleBarrelNavigableLayer_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include <vector>


/** A concrete NavigableLayer for the barrel 
 */

class SimpleBarrelNavigableLayer : public SimpleNavigableLayer {

public:

  SimpleBarrelNavigableLayer( BarrelDetLayer*, const BDLC&, 
			      const FDLC&, const FDLC&,const MagneticField* field, float);

  // NavigableLayer interface
  virtual vector<const DetLayer*> 
  nextLayers( PropagationDirection timeDirection) const;

  virtual vector<const DetLayer*> 
  nextLayers( const FreeTrajectoryState& fts, 
	      PropagationDirection timeDirection) const;

  // extended interface
  BDLC nextBarrelLayers()  { return theOuterBarrelLayers;}
  //  FDLC nextForwardLayers() { return theOuterForwardLayers;}

  virtual DetLayer* detLayer() const;
  virtual void   setDetLayer( DetLayer* dl);

  virtual void setInwardLinks(const BDLC&, const FDLC&);

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

  const BDLC& barrelLayers( const FreeTrajectoryState& fts,
			    PropagationDirection dir) const;

  const FDLC& forwardLayers( const FreeTrajectoryState& fts,
		       PropagationDirection dir) const;
};

#endif // SimpleBarrelNavigableLayer_H
