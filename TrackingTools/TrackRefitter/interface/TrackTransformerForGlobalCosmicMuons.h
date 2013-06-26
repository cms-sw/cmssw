#ifndef TrackingTools_TrackRefitter_TrackTransformerForGlobalCosmicMuons_H
#define TrackingTools_TrackRefitter_TrackTransformerForGlobalCosmicMuons_H

/** \class TrackTransformer
 *  This class takes a reco::Track and refits the rechits inside it.
 *  The final result is a Trajectory refitted and smoothed.
 *  To make the refitting (and the smoothing) the usual KF tools are used.
 *
 *  CAVEAT: till now (it will be changed in the near future) the class stores the
 *  pointers to the services, therefore EACH event the setServices(const edm::EventSetup&)
 *  method MUST be called in the code in which the TrackTransformer is used.
 *
 *  $Date: 2013/01/09 03:38:34 $
 *  $Revision: 1.5 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TrackRefitter/interface/TrackTransformerBase.h"

#include "TrackingTools/TrackRefitter/interface/RefitDirection.h"

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
class TrackerTopology;

class TrackTransformerForGlobalCosmicMuons: public TrackTransformerBase{

public:

  /// Constructor
  TrackTransformerForGlobalCosmicMuons(const edm::ParameterSet&);

  /// Destructor
  virtual ~TrackTransformerForGlobalCosmicMuons();
  
  // Operations

  /// Convert a reco::Track into Trajectory
  virtual std::vector<Trajectory> transform(const reco::Track&) const;

  /// the magnetic field
  const MagneticField* magneticField() const {return &*theMGField;}
  
  /// the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const {return theTrackingGeometry;}

  /// set the services needed by the TrackTransformer
  virtual void setServices(const edm::EventSetup&);

  /// the refitter used to refit the reco::Track
  edm::ESHandle<TrajectoryFitter> fitter(bool) const;
  
  /// the smoother used to smooth the trajectory which came from the refitting step
  edm::ESHandle<TrajectorySmoother> smoother(bool) const;

  TransientTrackingRecHit::ConstRecHitContainer
    getTransientRecHits(const reco::TransientTrack& track) const;

  /// check (via options) if this is a tracker rec hit for removal
  bool TrackerKeep(DetId id) const;
  /// check (via options) if this is a muon rec hit for removal
  bool MuonKeep(DetId id) const;
  
 protected:
  
 private:

  edm::ESHandle<Propagator> thePropagatorIO;
  edm::ESHandle<Propagator> thePropagatorOI;

  edm::ESHandle<Propagator> propagator(bool) const;

  
  unsigned long long theCacheId_TC;
  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_TRH;
  
  bool theRPCInTheFit;
  int	theSkipStationDT;
  int	theSkipStationCSC;
  int	theSkipWheelDT;
  int   theTrackerSkipSystem;
  int   theTrackerSkipSection;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;
  
  edm::ESHandle<TrajectoryFitter> theFitterIO;
  edm::ESHandle<TrajectoryFitter> theFitterOI;
  
  edm::ESHandle<TrajectorySmoother> theSmootherIO;
  edm::ESHandle<TrajectorySmoother> theSmootherOI;
 
  std::string theTrackerRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
  
  std::string theMuonRecHitBuilderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;
  
  const TrackerTopology *tTopo_;
};
#endif

