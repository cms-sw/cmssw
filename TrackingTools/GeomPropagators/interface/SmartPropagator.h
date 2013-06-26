#ifndef GeomPropagators_SmartPropagator_H
#define GeomPropagators_SmartPropagator_H

/** \class SmartPropagator
 *
 * A propagator which use different algorithm to propagate inside or outside
 * tracker
 *
 * \author  Stefano Lacaprara - INFN Padova 
 * \porting author Chang Liu - Purdue University 
 * $Date $
 * $Revision $
 *
 * Modification:
 *    26-Jun-2002 SL: theTkVolume is now a static
 *        ReferenceCountingPointer<BoundCylinder>
 *    28-Aug-2002 SL: added methods to unhide Propagator methods
 *    29-Oct-2002 SL: fixed clone and copy constructor, and BoundCylinder are
 *        build with CylinderBuilder to enforce the referencePointer
 *
 */

/* Collaborating Class Declarations */
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class Cylinder;
class Plane;


/* Class SmartPropagator Interface */

class SmartPropagator GCC11_FINAL : public Propagator {

  public:

    /* Constructor */ 
    ///Defines which propagator is used inside Tk and which outside
    SmartPropagator(Propagator* aTkProp, Propagator* aGenProp, const MagneticField* field,
        PropagationDirection dir = alongMomentum, float epsilon = 5) ;

    ///Defines which propagator is used inside Tk and which outside
    SmartPropagator(const Propagator& aTkProp, const Propagator& aGenProp,const MagneticField* field,
        PropagationDirection dir = alongMomentum, float epsilon = 5) ;

    ///Copy constructor
    SmartPropagator( const SmartPropagator& );

    /** virtual destructor */ 
    virtual ~SmartPropagator() ;

    ///Virtual constructor (using copy c'tor)
    virtual SmartPropagator* clone() const {
      return new SmartPropagator(getTkPropagator(),getGenPropagator(),magneticField());
    }

    ///setting the direction fo both components
    void setPropagationDirection (PropagationDirection dir) const
    {
      Propagator::setPropagationDirection (dir);
      getTkPropagator()->setPropagationDirection(dir);
      getGenPropagator()->setPropagationDirection(dir);
    }


    /* Operations as propagator*/ 
    TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                       const Surface& surface) const;

    TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& tsos, 
                                       const Surface& surface) const {
      return Propagator::propagate(tsos,surface);
    }

    TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts,
                                       const Plane& plane) const;

    TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& tsos,
                                       const Plane& plane) const {
      return Propagator::propagate(tsos, plane);
    }

    TrajectoryStateOnSurface propagate(const FreeTrajectoryState& fts, 
                                       const Cylinder& cylinder) const;

    TrajectoryStateOnSurface propagate(const TrajectoryStateOnSurface& tsos, 
                                       const Cylinder& cylinder) const {
      return Propagator::propagate(tsos, cylinder);
    }

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const FreeTrajectoryState& fts, 
                        const Surface& surface) const {
        return Propagator::propagateWithPath(fts,surface);
      }

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const TrajectoryStateOnSurface& tsos, 
                        const Surface& surface) const {
        return Propagator::propagateWithPath(tsos,surface);
      }

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const FreeTrajectoryState& fts, 
                        const Plane& plane) const;

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const TrajectoryStateOnSurface& tsos, 
                        const Plane& plane) const {
        return Propagator::propagateWithPath(tsos, plane);
      }

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const FreeTrajectoryState& fts, 
                        const Cylinder& cylinder) const;

    std::pair<TrajectoryStateOnSurface,double> 
      propagateWithPath(const TrajectoryStateOnSurface& tsos, 
                        const Cylinder& cylinder) const {
        return Propagator::propagateWithPath(tsos, cylinder);
      }

    ///true if a fts is inside tracker volume
    bool insideTkVol(const FreeTrajectoryState& fts) const ;
    ///true if a surface is inside tracker volume
    bool insideTkVol(const Surface& surface) const ;
    ///true if a cylinder is inside tracker volume
    bool insideTkVol(const Cylinder& cylin)  const ;
    ///true if a plane is inside tracker volume
    bool insideTkVol(const Plane& plane)  const ;

    ///return the propagator used inside tracker
    Propagator* getTkPropagator() const ;
    ///return the propagator used outside tracker
    Propagator* getGenPropagator() const ;
    ///return the magneticField
    virtual const MagneticField* magneticField() const {return theField;}

  private:
    ///build the tracker volume
  static void initTkVolume(float epsilon);

    mutable Propagator* theTkProp;
    mutable Propagator* theGenProp;
    const MagneticField* theField;
    static ReferenceCountingPointer<Cylinder> & theTkVolume();

  protected:

};

#endif // SMARTPROPAGATOR_H


