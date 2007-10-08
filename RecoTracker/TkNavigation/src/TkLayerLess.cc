#include "RecoTracker/TkNavigation/interface/TkLayerLess.h"
#include "Utilities/General/interface/CMSexception.h"

bool TkLayerLess::insideOutLess( const DetLayer* a, const DetLayer* b) const
{
  if (a == b) return false;

  const BarrelDetLayer* bla = 
    dynamic_cast<BarrelDetLayer*>(const_cast<DetLayer*>(a));
  const BarrelDetLayer* blb = 
    dynamic_cast<BarrelDetLayer*>(const_cast<DetLayer*>(b));

  if      ( bla!=0 && blb!=0) {  // barrel with barrel
    return bla->specificSurface().radius() < blb->specificSurface().radius();
  }

  const ForwardDetLayer* flb = 
    dynamic_cast<ForwardDetLayer*>(const_cast<DetLayer*>(b));

  if ( bla!=0 && flb!=0) {  // barrel with forward
    return barrelForwardLess( bla, flb);
  }

  const ForwardDetLayer* fla = 
    dynamic_cast<ForwardDetLayer*>(const_cast<DetLayer*>(a));

  if (fla!=0 && flb!=0) {  //  forward with forward
    return fabs( fla->position().z()) < fabs( flb->position().z());
  }
  if ( fla!=0 && blb!=0) {  // forward with barrel
    return !barrelForwardLess( blb, fla);
  }
  //throw DetLogicError("TkLayerLess: arguments are not Barrel or Forward DetLayers");
  throw Genexception("TkLayerLess: arguments are not Barrel or Forward DetLayers");

}

bool TkLayerLess::barrelForwardLess( const BarrelDetLayer* bla, 
				     const ForwardDetLayer* flb) const
{
  return bla->surface().bounds().length()/2. < fabs( flb->position().z());
}


