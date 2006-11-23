#ifndef TrackingTools_TrackRefitter_TrackTransformer_H
#define TrackingTools_TrackRefitter_TrackTransformer_H

/** \class TrackTransformer
 *  No description available.
 *
 *  $Date: 2006/11/22 18:36:45 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

namespace edm {class ParameterSet; class EventSetup;}
namespace reco {class Track; class TransientTrack;}

class TrajectoryFitter;
class TrajectorySmoother;
class TransientTrackingRecHitBuilder;
class Trajectory;
//class Propagator;

class TrackTransformer{

public:

  /// Constructor
  TrackTransformer(const edm::ParameterSet&);

  // Operations

  /// Convert a reco::Track into Trajectory
  std::vector<Trajectory> transform(const reco::Track&);

  /// Destructor
  virtual ~TrackTransformer();
  
  /// the magnetic field
  const MagneticField* magneticField() const {return &*theMGField;}
  
  /// the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const {return theTrackingGeometry;}

  /// set the services needed by the TrackTransformer
  void setServices(const edm::EventSetup& setup);

  /// the refitter used to refit the reco::Track
  edm::ESHandle<TrajectoryFitter> refitter() const {return theFitter;}
  
  /// the smoother used to smooth the trajectory which came from the refitting step
  edm::ESHandle<TrajectorySmoother> smoother() const {return theSmoother;}

 protected:
  
private:
  
  unsigned long long theCacheId_TC;
  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_TRH;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  
  std::string theFitterName;
  edm::ESHandle<TrajectoryFitter> theFitter;
  
  std::string theSmootherName;
  edm::ESHandle<TrajectorySmoother> theSmoother;

  TransientTrackingRecHit::ConstRecHitContainer
  getTransientRecHits(const reco::TransientTrack& track) const;
  
  std::string theTrackerRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;

  std::string theMuonRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;

  // std::string thePropagatorName;
  // edm::ESHandle<Propagator> thePropagator;
};
#endif

