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

class BeamHaloPropagator final : public Propagator {

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
    ~BeamHaloPropagator() override ;

    ///Virtual constructor (using copy c'tor)
    BeamHaloPropagator* clone() const override {
      return new BeamHaloPropagator(getEndCapTkPropagator(),getCrossTkPropagator(),magneticField(),propagationDirection());
    }


    void setPropagationDirection (PropagationDirection dir) override
    {
      Propagator::setPropagationDirection(dir);
      theEndCapTkProp->setPropagationDirection(dir);
      theCrossTkProp->setPropagationDirection(dir);
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


    ///true if the plane and the fts z position have different sign
      bool crossingTk(const FreeTrajectoryState& fts, const Plane& plane)  const ;

    ///return the propagator used in endcaps
    const Propagator* getEndCapTkPropagator() const ;
    ///return the propagator used to cross the tracker
    const Propagator* getCrossTkPropagator() const ;
    ///return the magneticField
    const MagneticField* magneticField() const override {return theField;}

  private:
    void directionCheck(PropagationDirection dir);

    Propagator* theEndCapTkProp;
    Propagator* theCrossTkProp;
    const MagneticField* theField;

  protected:

};

#endif


