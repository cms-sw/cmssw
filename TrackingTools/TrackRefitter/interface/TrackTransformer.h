#ifndef TrackingTools_TrackRefitter_TrackTransformer_H
#define TrackingTools_TrackRefitter_TrackTransformer_H

/** \class TrackTransformer
 *  This class takes a reco::Track and refits the rechits inside it.
 *  The final result is a Trajectory refitted and smoothed.
 *  To make the refitting (and the smoothing) the usual KF tools are used.
 *
 *  CAVEAT: till now (it will be changed in the near future) the class stores the
 *  pointers to the services, therefore EACH event the setServices(const edm::EventSetup&)
 *  method MUST be called in the code in which the TrackTransformer is used.
 *
 *  $Date: 2007/02/27 09:44:56 $
 *  $Revision: 1.6 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
namespace edm {class ParameterSet; class EventSetup;}
namespace reco {class TransientTrack;}

class TrajectoryFitter;
class TrajectorySmoother;
class Propagator;
class TransientTrackingRecHitBuilder;
class Trajectory;

class TrackTransformer{

public:

  /// Constructor
  TrackTransformer(const edm::ParameterSet&);

  /// Destructor
  virtual ~TrackTransformer();
  
  // Operations

  /// Convert a reco::Track into Trajectory
  std::vector<Trajectory> transform(const reco::Track&) const;

  /// Convert a reco::TrackRef into Trajectory
  std::vector<Trajectory> transform(const reco::TrackRef&) const;

  /// the magnetic field
  const MagneticField* magneticField() const {return &*theMGField;}
  
  /// the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const {return theTrackingGeometry;}

  /// set the services needed by the TrackTransformer
  void setServices(const edm::EventSetup&);

  /// the refitter used to refit the reco::Track
  edm::ESHandle<TrajectoryFitter> refitter() const {return theFitter;}
  
  /// the smoother used to smooth the trajectory which came from the refitting step
  edm::ESHandle<TrajectorySmoother> smoother() const {return theSmoother;}
  
 protected:
  
 private:
  std::string thePropagatorName;
  edm::ESHandle<Propagator> propagator() const {return thePropagator;}
  edm::ESHandle<Propagator> thePropagator;
  
  enum RefitDirection{insideOut,outsideIn,undetermined};
  
  unsigned long long theCacheId_TC;
  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_TRH;
  
  bool theRPCInTheFit;

  RefitDirection theRefitDirection;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  
  std::string theFitterName;
  edm::ESHandle<TrajectoryFitter> theFitter;
  
  std::string theSmootherName;
  edm::ESHandle<TrajectorySmoother> theSmoother;

  TransientTrackingRecHit::ConstRecHitContainer
    getTransientRecHits(const reco::TransientTrack& track) const;
  
  RefitDirection
    checkRecHitsOrdering(TransientTrackingRecHit::ConstRecHitContainer&) const;

  std::string theTrackerRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
  
  std::string theMuonRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
};
#endif

