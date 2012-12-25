#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

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

bool EtaPhiMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const Plane& plane) const {

  double dEta = fabs(tsos.globalPosition().eta() - plane.position().eta());
  double dPhi = deltaPhi< double > (tsos.globalPosition().phi(), plane.position().phi());

  LogDebug("EtaPhiMeasurementEstimator")<< "The state to compare with is \n"<< tsos << "\n"
					<< "The plane position center is: " << plane.position() << "\n"
					<< "the deta = " << thedEta << " --- the dPhi = " << thedPhi << "\n"
					<< "deta = "<< fabs(dEta)<< " --- dPhi = "<<fabs(dPhi);

  if (fabs(dEta) < thedEta && fabs(dPhi) <thedPhi)
    return true;
  else
    return false;
}

MeasurementEstimator::Local2DVector EtaPhiMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& tsos,
					const Plane& plane) const {

  return  Local2DVector(30., 30.);
}

