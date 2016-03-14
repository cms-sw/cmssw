#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/KalmanUpdators/interface/EtaPhiMeasurementEstimator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/Math/interface/deltaPhi.h"

std::pair<bool,double> 
EtaPhiMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const TrackingRecHit& aRecHit) const {

  auto dEta = std::abs(tsos.globalPosition().eta() - aRecHit.globalPosition().eta());
  auto dPhi = deltaPhi(tsos.globalPosition().barePhi(), aRecHit.globalPosition().barePhi());

  LogDebug("EtaPhiMeasurementEstimator")<< " The state to compare with is \n"<< tsos
					<< " The hit position is:\n" << aRecHit.globalPosition()
					<< " deta: "<< dEta<< " dPhi: "<<dPhi;

  if ( (dEta < thedEta) & (dPhi <thedPhi) )
    return std::make_pair(true, 1.0);
  else
    return std::make_pair(false, 0.0);
}

bool EtaPhiMeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				   const Plane& plane) const {

  auto dEta = std::abs(tsos.globalPosition().eta() - plane.eta());
  auto dPhi = deltaPhi(tsos.globalPosition().barePhi(), plane.phi());


  LogDebug("EtaPhiMeasurementEstimator")<< "The state to compare with is \n"<< tsos << "\n"
					<< "The plane position center is: " << plane.position() << "\n"
					<< "the deta = " << thedEta << " --- the dPhi = " << thedPhi << "\n"
					<< "deta = "<< fabs(dEta)<< " --- dPhi = "<<fabs(dPhi);

  return (std::abs(dEta) < thedEta) & (std::abs(dPhi) <thedPhi);
}

MeasurementEstimator::Local2DVector EtaPhiMeasurementEstimator::maximalLocalDisplacement( const TrajectoryStateOnSurface& tsos,
					const Plane& plane) const {

  return  Local2DVector(30., 30.);
}

