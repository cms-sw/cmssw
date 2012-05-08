#ifndef RKTestPropagator_H
#define RKTestPropagator_H

#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"
#include "TrackPropagation/RungeKutta/interface/RKPropagatorInS.h"

////////////////////////////////////////////////////////////////////////////
//
// Wrapper to test RKPropagatorInS in central volume using Veikko's 
// parametrized field... Martijn Mulders 29/6/07
//
////////////////////////////////////////////////////////////////////////////


class GlobalTrajectoryParameters;
class GlobalParametersWithPath;
class MagVolume;
class RKLocalFieldProvider;
class CartesianStateAdaptor;


class RKTestField GCC11_FINAL : public MagneticField
{
 public:
  virtual GlobalVector inTesla ( const GlobalPoint& ) const {return GlobalVector(0,0,4);}
};

class RKTestFieldProvider GCC11_FINAL : public MagneticFieldProvider<float> {
public:

RKTestFieldProvider (const MagneticField* field) : theField(field) {}

 virtual LocalVectorType valueInTesla( const LocalPointType& lp) const {
   // NOTE: the following transformation only works for the central volume
   // where global and local coordinates are numerically equal !
   GlobalPoint gp(lp.x(), lp.y(), lp.z());
   GlobalVector gv =  theField->inTesla(gp);
   return LocalVectorType(gv.x(),gv.y(),gv.z());
 }

 private:

 const MagneticField* theField; 

};

class RKTestMagVolume GCC11_FINAL : public MagVolume {
public:
  RKTestMagVolume( const PositionType& pos, const RotationType& rot, 
		       DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume( pos, rot, shape, mfp) {}

  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const {return true;}

  /// Access to volume faces - dummy implementation
  virtual const std::vector<VolumeSide>& faces() const {return theFaces;}

private:
  std::vector<VolumeSide> theFaces;
  
};

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class RKTestPropagator GCC11_FINAL : public Propagator {
 public:



  explicit RKTestPropagator( const MagneticField* field, PropagationDirection dir = alongMomentum,
  			     double tolerance = 5.e-5) :
    theTolerance(tolerance),
    theRealField(field),
    RKField(field),
    RKVol(RKTestMagVolume(MagVolume::PositionType(0,0,0), MagVolume::RotationType(),ddshapeless, &RKField ) ),
    theRKProp(new RKPropagatorInS(RKVol, dir, tolerance)) {}  

  virtual TrajectoryStateOnSurface 
    propagate (const FreeTrajectoryState& state, const Plane& plane) const {return theRKProp->propagate(state,plane);}
  
  virtual TrajectoryStateOnSurface 
    propagate (const FreeTrajectoryState& state, const Cylinder& cyl) const {
    return theRKProp->propagate(state,cyl);}

  virtual std::pair< TrajectoryStateOnSurface, double> 
    propagateWithPath (const FreeTrajectoryState& state, const Plane& plane) const {
    return theRKProp->propagateWithPath(state,plane);}

  virtual std::pair< TrajectoryStateOnSurface, double> 
    propagateWithPath (const FreeTrajectoryState& state, const Cylinder& cyl) const {return theRKProp->propagateWithPath(state,cyl);}

  TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& ts, 
                                     const Plane& plane) const {return theRKProp->propagate(ts,plane);}

  virtual void setPropagationDirection(PropagationDirection dir) const {
    theRKProp->setPropagationDirection(dir);
  }

  virtual PropagationDirection propagationDirection() const {
    return theRKProp->propagationDirection();
  }
  

  Propagator* clone() const
    {

      return new RKTestPropagator(magneticField(),propagationDirection(),theTolerance);

    }

  virtual const MagneticField* magneticField() const { return theRealField;}


 private:
  float theTolerance;
  const MagneticField* theRealField;
  RKTestFieldProvider RKField;
  RKTestMagVolume  RKVol;
  DeepCopyPointerByClone<Propagator> theRKProp;  
};

#endif
