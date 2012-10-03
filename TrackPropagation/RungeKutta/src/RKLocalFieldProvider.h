#ifndef RKLocalFieldProvider_H
#define RKLocalFieldProvider_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

class MagVolume;

class dso_internal RKLocalFieldProvider {
public:

    typedef GloballyPositioned<float>            Frame;
    typedef Frame::GlobalVector                  GlobalVector;
    typedef Frame::GlobalPoint                   GlobalPoint;
    typedef Frame::LocalVector                   LocalVector;
    typedef Frame::LocalPoint                    LocalPoint;
    typedef Frame::PositionType                  Position;
    typedef Frame::RotationType                  Rotation;
    typedef GlobalVector::BasicVectorType        Vector;

    /// Local field access to the MagVolume field, in the MagVolume frame
    explicit RKLocalFieldProvider( const MagVolume& vol);

    /// Local field access to the MagVolume field, transformed to the "frame" frame
    RKLocalFieldProvider( const MagVolume& vol, const Frame& frame);

    /// the argument lp is in the local frame specified in the constructor
    Vector inTesla( const LocalPoint& lp) const;

    Vector inTesla( double x, double y, double z) const {
	return inTesla( LocalPoint(x,y,z));
    }

    Vector inTesla( const Vector& v) const {
	return inTesla( LocalPoint(v));
    }

    /// The reference frame in which the field is defined
    const Frame& frame() const {return theFrame;}

private:

    const MagVolume & theVolume;
    const Frame&     theFrame;
    bool             transform_;


};

#endif
