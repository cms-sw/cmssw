#ifndef TrackAssociatorByPosition_h
#define TrackAssociatorByPosition_h

/** \class TrackAssociatorByPosition
 *  Class that performs the association of reco::Tracks and TrackingParticles based on position in muon detector
 *
 *  $Date: 2007/07/23 19:09:55 $
 *  $Revision: 1.2 $
 *  \author vlimant
 */

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>

#include<map>

//Note that the Association Map is filled with -ch2 and not chi2 because it is ordered using std::greater:
//the track with the lowest association chi2 will be the first in the output map.

class TrackAssociatorByPosition : public TrackAssociatorBase {

 public:

  /// Constructor with propagator and PSet
   TrackAssociatorByPosition(const edm::ParameterSet& iConfig,
			     const TrackingGeometry * geo, 
			     const Propagator * prop){
     theGeometry = geo;
     thePropagator = prop;
     theMinIfNoMatch = iConfig.getParameter<bool>("MinIfNoMatch");
     theQminCut = iConfig.getParameter<double>("QminCut");
     theQCut = iConfig.getParameter<double>("QCut");
     thePositionMinimumDistance = iConfig.getParameter<double>("positionMinimumDistance");
     std::string  meth= iConfig.getParameter<std::string>("method");
     if (meth=="chi2"){ theMethod =0; }
     else if (meth=="dist"){theMethod =1;}
     else if (meth=="momdr"){theMethod = 2;}
     else if (meth=="posdr"){theMethod = 3;}
     else{
       edm::LogError("TrackAssociatorByPosition")<<meth<<" mothed not recognized. Use dr or chi2.";     }
   };


  /// Destructor
  ~TrackAssociatorByPosition(){
  };


  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  reco::RecoToSimCollection associateRecoToSim (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>&, 
						const edm::Event * event = 0) const;

  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  reco::SimToRecoCollection associateSimToReco (edm::Handle<reco::TrackCollection>&, 
						edm::Handle<TrackingParticleCollection>& ,
						const edm::Event * event = 0) const;

  double quality(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface &)const;

 private:

  const TrackingGeometry * theGeometry;
  const Propagator * thePropagator;
  uint theMethod;
  double theQminCut;
  double theQCut;
  bool theMinIfNoMatch;
  double thePositionMinimumDistance;
 
  FreeTrajectoryState getState(const reco::Track &) const;
  TrajectoryStateOnSurface getState(const TrackingParticle &)const;

};

#endif
