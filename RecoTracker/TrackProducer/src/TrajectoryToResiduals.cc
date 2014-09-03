#include "TrajectoryToResiduals.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

reco::TrackResiduals trajectoryToResiduals (const Trajectory &trajectory,
					    enum reco::TrackResiduals::ResidualType type) 
{
     reco::TrackResiduals residuals(type);
     int i_residual = 0;
     Trajectory::DataContainer::const_iterator i_fwd = 
	  trajectory.measurements().begin(); 
     Trajectory::DataContainer::const_reverse_iterator i_bwd = 
	  trajectory.measurements().rbegin(); 
     Trajectory::DataContainer::const_iterator i_end = 
	  trajectory.measurements().end(); 
     Trajectory::DataContainer::const_reverse_iterator i_rend = 
	  trajectory.measurements().rend(); 
     bool forward = trajectory.direction() == alongMomentum;
     for (; forward ? i_fwd != i_end : i_bwd != i_rend; 
	  ++i_fwd, ++i_bwd, ++i_residual) {
	  const TrajectoryMeasurement *i = forward ? &*i_fwd : &*i_bwd;
	  if (!i->recHit()->isValid()||i->recHit()->det()==0) 
	       continue;
	  TrajectoryStateCombiner combine;
	  if (!i->forwardPredictedState().isValid() || !i->backwardPredictedState().isValid())
	    {
	      edm::LogError("InvalideState")<<"one of the step is invalid";
	      continue;
	    }

	  TrajectoryStateOnSurface && combo = combine(i->forwardPredictedState(),
       						   i->backwardPredictedState());
	  
	  if (!combo.isValid()){
	    edm::LogError("InvalideState")<<"the combined state is invalid";
	    continue;
	  }

	  LocalPoint && combo_localpos = combo.localPosition();
	  LocalError && combo_localerr = combo.localError().positionError();
	  LocalPoint && dethit_localpos = i->recHit()->localPosition();     
	  LocalError && dethit_localerr = i->recHit()->localPositionError();
	  auto const &  error_including_alignment = dethit_localerr; // align error nwo is included 
	  switch (type) {
	  case reco::TrackResiduals::X_Y_RESIDUALS: 
	  {
	       auto x = (dethit_localpos.x() - combo_localpos.x()) / 
		    std::sqrt(error_including_alignment.xx());
	       auto y = (dethit_localpos.y() - combo_localpos.y()) / 
		    std::sqrt(error_including_alignment.yy());
	       residuals.setResidualXY(i_residual, x, y);
	       break;
	  }
	  case reco::TrackResiduals::X_Y_PULLS:
	  {
	       auto x = (dethit_localpos.x() - combo_localpos.x()) / 
		    std::sqrt(error_including_alignment.xx() + combo_localerr.xx());
	       auto y = (dethit_localpos.y() - combo_localpos.y()) / 
		    std::sqrt(error_including_alignment.yy() + combo_localerr.yy());
	       residuals.setPullXY(i_residual, x, y);
	       break;
	  }
	  default:
	       assert(0);
	  }
     }
     return residuals;
}
