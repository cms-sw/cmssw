#ifndef ChargeSignificanceTrajectoryFilter_H
#define ChargeSignificanceTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TempTrajectory.h"

/** A TrajectoryFilter that stops reconstruction if P_t drops
 *  below some value at some confidence level.
 *  The CkfTrajectoryBuilder uses this class to
 *  implement the minimal P_t cut.
 */

class ChargeSignificanceTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit ChargeSignificanceTrajectoryFilter(float qsig):  theChargeSignificance(qsig) {}

  explicit ChargeSignificanceTrajectoryFilter( const edm::ParameterSet & pset, edm::ConsumesCollector& iC):
    theChargeSignificance(pset.getParameter<double>("chargeSignificance")) {}

  bool qualityFilter( const Trajectory& traj) const override { return traj.isValid();}
  bool qualityFilter( const TempTrajectory& traj)const override { return traj.isValid();}
 
  bool toBeContinued( Trajectory& traj)const override { return TBC<Trajectory>(traj);}
  bool toBeContinued( TempTrajectory& traj) const override { return TBC<TempTrajectory>(traj);}
  
  std::string name() const override {return "ChargeSignificanceTrajectoryFilter";}

protected:

  template <class T> bool TBC(T & traj) const{
    const typename T::DataContainer & tms = traj.measurements();
    // Check flip in q-significance. The loop over all TMs could be 
    // avoided by storing the current significant q in the trajectory

    if ( theChargeSignificance>0. ) {      

      float qSig = 0;

      // skip first two hits (don't rely on significance of q/p)
      for( typename T::DataContainer::size_type itm=2; itm<tms.size(); ++itm ) {
	TrajectoryStateOnSurface const & tsos = tms[itm].updatedState();
	if ( !tsos.isValid() )  continue;

        auto significance = tsos.localParameters().vector()(0) /
	  std::sqrt(float(tsos.localError().matrix()(0,0)));

	// don't deal with measurements compatible with 0
	if ( std::abs(significance)<theChargeSignificance )  continue;

	//
	// if charge not yet defined: store first significant Q
	//
	if ( qSig==0 ) qSig = significance;
    
	//
	//  invalidate and terminate in case of a change of sign
	//
	if (significance*qSig<0) {
	    traj.invalidate();
            traj.setStopReason(StopReason::CHARGE_SIGNIFICANCE);
	    return false;
	}
      }
    }
    return true;
  }

  float theChargeSignificance;

};

#endif
