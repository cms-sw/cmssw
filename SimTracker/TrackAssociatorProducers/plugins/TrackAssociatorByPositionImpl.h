#ifndef TrackAssociatorByPositionImpl_h
#define TrackAssociatorByPositionImpl_h

/** \class TrackAssociatorByPositionImpl
 *  Class that performs the association of reco::Tracks and TrackingParticles based on position in muon detector
 *
 *  \author vlimant
 */

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include <map>

//Note that the Association Map is filled with -ch2 and not chi2 because it is ordered using std::greater:
//the track with the lowest association chi2 will be the first in the output map.

class TrackAssociatorByPositionImpl : public reco::TrackToTrackingParticleAssociatorBaseImpl {
public:
  typedef std::pair<TrackingParticleRef, TrackPSimHitRef> SimHitTPPair;
  typedef std::vector<SimHitTPPair> SimHitTPAssociationList;
  enum class Method { chi2, dist, momdr, posdr };

  TrackAssociatorByPositionImpl(edm::EDProductGetter const& productGetter,
                                const TrackingGeometry* geo,
                                const Propagator* prop,
                                const SimHitTPAssociationList* assocList,
                                double qMinCut,
                                double qCut,
                                double positionMinimumDistance,
                                Method method,
                                bool minIfNoMatch,
                                bool considerAllSimHits)
      : productGetter_(&productGetter),
        theGeometry(geo),
        thePropagator(prop),
        theSimHitsTPAssoc(assocList),
        theQminCut(qMinCut),
        theQCut(qCut),
        thePositionMinimumDistance(positionMinimumDistance),
        theMethod(method),
        theMinIfNoMatch(minIfNoMatch),
        theConsiderAllSimHits(considerAllSimHits) {}

  /// compare reco to sim the handle of reco::Track and TrackingParticle collections

  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
                                               const edm::RefVector<TrackingParticleCollection>&) const override;

  /// compare reco to sim the handle of reco::Track and TrackingParticle collections

  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
                                               const edm::RefVector<TrackingParticleCollection>&) const override;

private:
  double quality(const TrajectoryStateOnSurface&, const TrajectoryStateOnSurface&) const;

  edm::EDProductGetter const* productGetter_;
  const TrackingGeometry* theGeometry;
  const Propagator* thePropagator;
  const SimHitTPAssociationList* theSimHitsTPAssoc;
  double theQminCut;
  double theQCut;
  double thePositionMinimumDistance;
  Method theMethod;
  bool theMinIfNoMatch;
  bool theConsiderAllSimHits;

  FreeTrajectoryState getState(const reco::Track&) const;
  TrajectoryStateOnSurface getState(const TrackingParticleRef&, const SimHitTPAssociationList& simHitsTPAssoc) const;
  //edm::InputTag _simHitTpMapTag;
};

#endif
