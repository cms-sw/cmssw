#include "RKLocalFieldProvider.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

RKLocalFieldProvider::RKLocalFieldProvider( const MagVolume& vol) :
  theVolume( &vol), theFrame(vol), transform_(false) {}

RKLocalFieldProvider::RKLocalFieldProvider( const MagVolume& vol, const Frame& frame) :
  theVolume( &vol), theFrame(frame), transform_(true) {}

/*
RKLocalFieldProvider::RKLocalFieldProvider() : 
  theVolume(0), theFrame(globalFrame()), transform_(false) {}

RKLocalFieldProvider::RKLocalFieldProvider( const Frame& frame) :
  theVolume(0), theFrame(frame), transform_(true) {}
*/

RKLocalFieldProvider::Vector RKLocalFieldProvider::inTesla( const LocalPoint& lp) const 
{
  if (theVolume != 0) {
    if (transform_) {
      LocalPoint vlp( theVolume->toLocal( theFrame.toGlobal( lp)));
      return theFrame.toLocal( theVolume->toGlobal( theVolume->fieldInTesla( vlp))).basicVector();
    }
    else {
      return theVolume->fieldInTesla( lp).basicVector();
    }
  }
  else {
    /*
    if (transform_) {
      GlobalVector gv( MagneticField::inTesla(theFrame.toGlobal(lp)));
      return theFrame.toLocal(gv).basicVector();
    }
    else {
      // the "local" frame is actually global
      return MagneticField::inTesla( GlobalPoint( lp.basicVector())).basicVector();
    }
    */

    return RKLocalFieldProvider::Vector(0,0,0);
  }
}

