#ifndef ShiftingMagneticFieldProvider_H
#define ShiftingMagneticFieldProvider_H

#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"
#include "MagneticField/VolumeGeometry/interface/MagVolume.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


template <class T>
class ShiftingMagneticFieldProvider GCC11_FINAL : public MagneticFieldProvider<T> {
public:

  typedef typename MagneticFieldProvider<T>::LocalPointType    LocalPointType;
  typedef typename MagneticFieldProvider<T>::LocalVectorType   LocalVectorType;
  typedef typename MagneticFieldProvider<T>::GlobalPointType   GlobalPointType;
  typedef typename MagneticFieldProvider<T>::GlobalVectorType  GlobalVectorType;


  ShiftingMagneticFieldProvider( const MagVolume& magvol, 
				 const MagVolume::PositionType& pos,
				 const MagVolume::RotationType& rot);

  virtual LocalVectorType valueInTesla( const LocalPointType& p) const;

private:

  enum FrameRelation {sameFrame, sameOrientation, differentFrames};

  GloballyPositioned<T>     theFrame;
  FrameRelation             theFrameRelation;
  const MagVolume&          theMagVolume;
  Basic3DVector<T>          theShift;

};

#include "TrackPropagation/NavGeometry/src/ShiftingMagneticFieldProvider.icc"

#endif
