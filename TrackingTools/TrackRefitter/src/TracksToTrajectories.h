#ifndef TrackingTools_TrackRefitter_TracksToTrajectories_H
#define TrackingTools_TrackRefitter_TracksToTrajectories_H

/** \class TracksToTrajectories
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco {class TransientTrack;}

class TrajectoryFitter;
class TrajectorySmoother;
class TransientTrackingRecHitBuilder;
class Propagator;
class TracksToTrajectories: public edm::EDProducer{
public:

  /// Constructor
  TracksToTrajectories(const edm::ParameterSet&);

  // Operations

  /// Convert Tracks into Trajectories
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Destructor
  virtual ~TracksToTrajectories();
  
protected:
  
private:
  /// get the magnetic field
  edm::ESHandle<MagneticField> magneticField() const {return theMGField;}
  
  /// get the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const {return theTrackingGeometry;}

  void extractServices(const edm::EventSetup& setup);
  unsigned long long theCacheId_TC;
  unsigned long long theCacheId_GTG;
  unsigned long long theCacheId_MG;
  unsigned long long theCacheId_TRH;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;

  edm::InputTag theTracksLabel;
  
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

  std::string thePropagatorName;
  edm::ESHandle<Propagator> thePropagator;
};
#endif

