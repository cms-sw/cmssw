#ifndef GeomPropagators_SmartPropagator_H
#define GeomPropagators_SmartPropagator_H

/** \class SmartPropagator
 *
 * A propagator which use different algorithm to propagate inside or outside
 * tracker
 *
 * \author  Stefano Lacaprara - INFN Padova
 * \porting author Chang Liu - Purdue University
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

class SmartPropagator final : public Propagator {

  public:

    /* Constructor */
    ///Defines which propagator is used inside Tk and which outside
    SmartPropagator(const Propagator* aTkProp, const Propagator* aGenProp, const MagneticField* field,
        PropagationDirection dir = alongMomentum, float epsilon = 5) ;

    ///Defines which propagator is used inside Tk and which outside
    SmartPropagator(const Propagator& aTkProp, const Propagator& aGenProp,const MagneticField* field,
        PropagationDirection dir = alongMomentum, float epsilon = 5) ;

    ///Copy constructor
    SmartPropagator( const SmartPropagator& );

    /** virtual destructor */
    ~SmartPropagator() override ;

    ///Virtual constructor (using copy c'tor)
    SmartPropagator* clone() const override {
      return new SmartPropagator(getTkPropagator(),getGenPropagator(),magneticField());
    }

    ///setting the direction fo both components
    void setPropagationDirection (PropagationDirection dir) override
    {
      Propagator::setPropagationDirection (dir);
      theTkProp->setPropagationDirection(dir);
      theGenProp->setPropagationDirection(dir);
    }


    using Propagator::propagate;
    using Propagator::propagateWithPath;



 private:
    std::pair<TrajectoryStateOnSurface,double>
      propagateWithPath(const FreeTrajectoryState& fts,
                        const Plane& plane) const override;
    
    
    std::pair<TrajectoryStateOnSurface,double>
      propagateWithPath(const FreeTrajectoryState& fts,
                        const Cylinder& cylinder) const override;
    
    std::pair< TrajectoryStateOnSurface, double>
      propagateWithPath (const TrajectoryStateOnSurface& tsos, const Plane& sur) const override;
    
    std::pair< TrajectoryStateOnSurface, double>
      propagateWithPath (const TrajectoryStateOnSurface& tsos, const Cylinder& sur) const override;

 public:

    ///true if a fts is inside tracker volume
    bool insideTkVol(const FreeTrajectoryState& fts) const ;
    ///true if a surface is inside tracker volume
    bool insideTkVol(const Surface& surface) const ;
    ///true if a cylinder is inside tracker volume
    bool insideTkVol(const Cylinder& cylin)  const ;
    ///true if a plane is inside tracker volume
    bool insideTkVol(const Plane& plane)  const ;

    ///return the propagator used inside tracker
    const Propagator* getTkPropagator() const ;
    ///return the propagator used outside tracker
    const Propagator* getGenPropagator() const ;
    ///return the magneticField
    const MagneticField* magneticField() const override {return theField;}

  private:
    ///build the tracker volume
    void initTkVolume(float epsilon);

    Propagator* theTkProp;
    Propagator* theGenProp;
    const MagneticField* theField;
    ReferenceCountingPointer<Cylinder> theTkVolume;

  protected:

};

#endif // SMARTPROPAGATOR_H


