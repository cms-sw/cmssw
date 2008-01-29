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


class RKTestField : public MagneticField
{
 public:
  virtual GlobalVector inTesla ( const GlobalPoint& ) const {return GlobalVector(0,0,4);}
};

class RKTestFieldProvider : public MagneticFieldProvider<float> {
public:
  virtual LocalVectorType valueInTesla( const LocalPointType& gp) const {
    
//
//    B-field in Tracker volume
//    
//     In:   xyz[3]: coordinates (m)
//    Out:  bxyz[3]: Bx,By,Bz    (kG)
//
//    Valid for r<1.2 and |z|<3.0               V.Karimäki 040301
//                                 Updated for CMSSW field 070424
//

// b0=field at centre, l=solenoid length, a=radius (m) (phenomen. parameters) 

    static const float b0=40.681, l=15.284, a=4.6430;   // cmssw
    static float ap2=4.0*a*a/(l*l);  
    static float hb0=0.5*b0*sqrt(1.0+ap2);
    static float hlova=1.0/sqrt(ap2);
    static float ainv=2.0*hlova/l;
    float xyz[3];//, bxyz[3];
    xyz[0]=0.01*gp.x();
    xyz[1]=0.01*gp.y();
    xyz[2]=0.01*gp.z();
    
    float r=sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
    float z=xyz[2];
    float az = fabs(z);
    if (r<1.2&&az<3.0) {
      float zainv=z*ainv;
      float rinv=(r>0.0) ? 1.0/r : 0.0;
      float u=hlova-zainv;
      float v=hlova+zainv;
      float fu[5],gv[5];

      float a,b;
      //      ffunkti(u,fu);
      a=1.0/(1.0+u*u);
      b=sqrt(a);
      fu[0]=u*b;
      fu[1]=a*b;
      fu[2]=-3.0*u*a*fu[1];
      fu[3]=a*fu[2]*((1.0/u)-4.0*u);

      //      ffunkti(v,gv);
      a=1.0/(1.0+v*v);
      b=sqrt(a);
      gv[0]=v*b;
      gv[1]=a*b;
      gv[2]=-3.0*v*a*gv[1];
      gv[3]=a*gv[2]*((1.0/v)-4.0*v);

      float rat=r*ainv;
      float corrr=0.00894*r*z*(az-2.2221)*(az-2.2221);
      float corrz=-0.02996*exp(-0.5*(az-1.9820)*(az-1.9820)/(0.78915*0.78915));
      float br=hb0*0.5*rat*(fu[1]-gv[1]-0.125*(fu[3]-gv[3])*rat*rat)+corrr;
      float bz=hb0*(fu[0]+gv[0]-(fu[2]+gv[2])*0.25*rat*rat)+corrz;

      LocalVectorType bvec = 0.1*LocalVectorType(br*xyz[0]*rinv, br*xyz[1]*rinv, bz);
      //      //      std::cout << "Returning local B value " << bvec << std::endl;
      return bvec;
    }   
    return LocalVectorType();

  } 
};

class RKTestMagVolume : public MagVolume {
public:
  RKTestMagVolume( const PositionType& pos, const RotationType& rot, 
		       DDSolidShape shape, const MagneticFieldProvider<float> * mfp) :
    MagVolume( pos, rot, shape, mfp) {}

  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const {return true;}

  /// Access to volume faces
  virtual std::vector<VolumeSide> faces() const {return std::vector<VolumeSide>();}
  
};

#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class RKTestPropagator : public Propagator {
 public:

  explicit RKTestPropagator( const MagneticField* dummyfield, PropagationDirection dir = alongMomentum,
  			     double tolerance = 5.e-5) :
    RKField ( RKTestFieldProvider() ),
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

  virtual void setPropagationDirection(PropagationDirection dir) {
    theRKProp->setPropagationDirection(dir);
  }

  virtual PropagationDirection propagationDirection() const {
    return theRKProp->propagationDirection();
  }
  

  Propagator* clone() const
    {
      return new RKTestPropagator(*this);
    }

  virtual const MagneticField* magneticField() const { return theRKProp->magneticField();}


 private:
  RKTestFieldProvider RKField;
  RKTestMagVolume  RKVol;
  DeepCopyPointerByClone<Propagator> theRKProp;  
};

#endif
