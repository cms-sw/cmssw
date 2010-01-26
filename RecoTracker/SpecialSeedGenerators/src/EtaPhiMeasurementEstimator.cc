#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "../interface/EtaPhiMeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "DataFormats/Math/interface/deltaPhi.h"

std::pair<bool,double> 
EtaPhiMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TransientTrackingRecHit& aRecHit) const {

  double dEta = fabs(tsos.globalPosition().eta() - aRecHit.globalPosition().eta());
  double dPhi = deltaPhi< double > (tsos.globalPosition().phi(), aRecHit.globalPosition().phi());

  LogDebug("EtaPhiMeasurementEstimator")<< " The state to compare with is \n"<< tsos
					<< " The hit position is:\n" << aRecHit.globalPosition()
					<< " deta: "<< dEta<< " dPhi: "<<dPhi;

  if (dEta < thedEta && dPhi <thedPhi)
    return std::make_pair(true, 1.0);
  else
    return std::make_pair(false, 0.0);
}


