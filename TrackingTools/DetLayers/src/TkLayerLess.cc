#include "TrackingTools/DetLayers/interface/TkLayerLess.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool TkLayerLess::insideOutLess( const DetLayer* a, const DetLayer* b) const
{
  if (a == b) return false;

  const BarrelDetLayer* bla = 
    dynamic_cast<const BarrelDetLayer*>(a);
  const BarrelDetLayer* blb = 
    dynamic_cast<const BarrelDetLayer*>(b);

  if      ( bla!=nullptr && blb!=nullptr) {  // barrel with barrel
    return bla->specificSurface().radius() < blb->specificSurface().radius();
  }

  const ForwardDetLayer* flb = 
    dynamic_cast<const ForwardDetLayer*>(b);

  if ( bla!=nullptr && flb!=nullptr) {  // barrel with forward
    return barrelForwardLess( bla, flb);
  }

  const ForwardDetLayer* fla = 
    dynamic_cast<const ForwardDetLayer*>(a);

  if (fla!=nullptr && flb!=nullptr) {  //  forward with forward
    return fabs( fla->position().z()) < fabs( flb->position().z());
  }
  if ( fla!=nullptr && blb!=nullptr) {  // forward with barrel
    return !barrelForwardLess( blb, fla);
  }
  //throw DetLogicError("TkLayerLess: arguments are not Barrel or Forward DetLayers");
  throw cms::Exception("TkLayerLess", "Arguments are not Barrel or Forward DetLayers");

}

bool TkLayerLess::barrelForwardLess( const BarrelDetLayer* bla, 
				     const ForwardDetLayer* flb) const
{
  return bla->surface().bounds().length()/2. < fabs( flb->position().z());
}


bool TkLayerLess::insideOutLessSigned( const DetLayer* a, const DetLayer* b) const
{
  if (a == b) return false;

  const BarrelDetLayer* bla =
    dynamic_cast<const BarrelDetLayer*>(a);
  const BarrelDetLayer* blb =
    dynamic_cast<const BarrelDetLayer*>(b);

  if      ( bla!=nullptr && blb!=nullptr) {  // barrel with barrel
    return bla->specificSurface().radius() < blb->specificSurface().radius();
  }

  const ForwardDetLayer* flb =
    dynamic_cast<const ForwardDetLayer*>(b);

  if ( bla!=nullptr && flb!=nullptr) {  // barrel with forward
    return barrelForwardLess( bla, flb);
  }

  const ForwardDetLayer* fla =
    dynamic_cast<const ForwardDetLayer*>(a);

  if (fla!=nullptr && flb!=nullptr) {  //  forward with forward
    if (fla->position().z()*flb->position().z() > 0) {// same z-sign
      //regular ordering when same sign
      LogDebug("BeamHaloTkLayerLess")<<"reaching this: "
				     <<theFromLayerSign<<" "
				     <<fla->position().z()<<" "
				     <<flb->position().z();
      return (fabs(fla->position().z()) < fabs( flb->position().z()));
    }
    else{//layers compared are not on the same z-side
      LogDebug("BeamHaloTkLayerLess")<<"reaching this at least: "
				     <<theFromLayerSign<<" "
				     <<fla->position().z()<<" "
				     <<flb->position().z();

      if (theFromLayerSign*fla->position().z()>0){
	//"fla" and original layer are on the same side
	//say that fla is less than flb
	return false;
      }else{
	return true;
      }
    }
  }
  if ( fla!=nullptr && blb!=nullptr) {  // forward with barrel
    return !barrelForwardLess( blb, fla);
  }
  throw cms::Exception("BeamHaloTkLayerLess", "Arguments are not Barrel or Forward DetLayers");

}


