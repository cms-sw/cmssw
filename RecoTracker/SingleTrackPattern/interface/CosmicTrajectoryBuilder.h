#ifndef CosmicTrajectoryBuilder_h
#define CosmicTrajectoryBuilder_h

//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
class TrajectoryMeasurement;
class CosmicTrajectoryBuilder 
{

  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;

 public:
  
  CosmicTrajectoryBuilder(const edm::ParameterSet& conf);
  ~CosmicTrajectoryBuilder();

  /// Runs the algorithm
    void run(const TrajectorySeedCollection &collseed,
	     const SiStripRecHit2DLocalPosCollection &collstereo,
	     const SiStripRecHit2DLocalPosCollection &collrphi ,
	     const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
	     const SiPixelRecHitCollection &collpixel,
	     const edm::EventSetup& es,
	     TrackCandidateCollection &output,
	     const TrackerGeometry& tracker);
    void init(const edm::EventSetup& es);
 private:
    std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed,
							const TrackerGeometry& tracker) const;
    Trajectory createStartingTrajectory( const TrajectorySeed& seed,
					 const TrackerGeometry& tracker) const;
 private:
    //   const MagneticField*     magfield;
  edm::ESHandle<MagneticField> magfield;
    edm::ParameterSet conf_;
    TrajectoryStateTransform tsTransform;
    AnalyticalPropagator  *thePropagator;


};

#endif
