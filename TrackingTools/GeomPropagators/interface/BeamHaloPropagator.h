#ifndef GeomPropagators_BeamHaloPropagator_H
#define GeomPropagators_BeamHaloPropagator_H

/** \class BeamHaloPropagator
 *
 * A propagator which use different algorithm to propagate
 * within an endcap or to cross over to the other endcap
 *
 * \author  Jean-Roch VLIMANT UCSB

 *
 */

/* Collaborating Class Declarations */
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

/* Class BeamHaloPropagator Interface */

class BeamHaloPropagator GCC11_FINAL : public Propagator {

  public:

    /* Constructor */
    ///Defines which propagator is used inside endcap and in barrel
    BeamHaloPropagator(const Propagator* aEndCapTkProp, const Propagator* aCrossTkProp, const MagneticField* field,
		       PropagationDirection dir = alongMomentum);

    ///Defines which propagator is used inside endcap and in barrel
    BeamHaloPropagator(const Propagator& aEndCapTkProp,const Propagator& aCrossTkProp, const MagneticField* field,
		       PropagationDirection dir = alongMomentum);


    ///Copy constructor
    BeamHaloPropagator( const BeamHaloPropagator& );

    /** virtual destructor */
    virtual ~BeamHaloPropagator() ;

    ///Virtual constructor (using copy c'tor)
    virtual BeamHaloPropagator* clone() const {
      return new BeamHaloPropagator(getEndCapTkPropagator(),getCrossTkPropagator(),magneticField(),propagationDirection());
    }


    void setPropagationDirection (PropagationDirection dir) override
    {
      Propagator::setPropagationDirection(dir);
      theEndCapTkProp->setPropagationDirection(dir);
      theCrossTkProp->setPropagationDirection(dir);
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

    ///true if the plane and the fts z position have different sign
      bool crossingTk(const FreeTrajectoryState& fts, const Plane& plane)  const ;

    ///return the propagator used in endcaps
    const Propagator* getEndCapTkPropagator() const ;
    ///return the propagator used to cross the tracker
    const Propagator* getCrossTkPropagator() const ;
    ///return the magneticField
    virtual const MagneticField* magneticField() const {return theField;}

  private:
    void directionCheck(PropagationDirection dir);

    Propagator* theEndCapTkProp;
    Propagator* theCrossTkProp;
    const MagneticField* theField;

  protected:

};

#endif


