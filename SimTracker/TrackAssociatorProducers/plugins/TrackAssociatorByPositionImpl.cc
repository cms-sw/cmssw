#include "TrackAssociatorByPositionImpl.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include "DataFormats/Math/interface/deltaR.h"
//#include "PhysicsTools/Utilities/interface/DeltaR.h"

using namespace edm;
using namespace reco;


TrajectoryStateOnSurface TrackAssociatorByPositionImpl::getState(const TrackingParticleRef& st, const SimHitTPAssociationList& simHitsTPAssoc)const{

  using SimHitTPPair =  std::pair<TrackingParticleRef, TrackPSimHitRef>;
  SimHitTPPair clusterTPpairWithDummyTP(st,TrackPSimHitRef());//SimHit is dummy:
  // sorting only the cluster is needed
  auto range = std::equal_range(simHitsTPAssoc.begin(), simHitsTPAssoc.end(),
				clusterTPpairWithDummyTP, [](const SimHitTPPair& iLHS, const SimHitTPPair& iRHS) -> bool {
                                  return iLHS.first.key()> iRHS.first.key();
                                });
  
  //  TrackingParticle* simtrack = const_cast<TrackingParticle*>(&st);
  //loop over PSimHits
  const PSimHit * psimhit=0;
  const BoundPlane * plane=0;
  double dLim=thePositionMinimumDistance;

  //    look for the further most hit beyond a certain limit
  auto start=range.first;
  auto end=range.second;
  LogDebug("TrackAssociatorByPositionImpl")<<range.second-range.first<<" PSimHits.";

  unsigned int count=0;
  for (auto ip=start;ip!=end;++ip){    

    TrackPSimHitRef psit = ip->second;

    //get the detid
    DetId dd(psit->detUnitId());

    if (!theConsiderAllSimHits && dd.det()!=DetId::Tracker) continue; 

    LogDebug("TrackAssociatorByPositionImpl")<<count++<<"] PSimHit on: "<<dd.rawId();
    //get the surface from the global geometry
    const GeomDet * gd=theGeometry->idToDet(dd);
    if (!gd){edm::LogError("TrackAssociatorByPositionImpl")<<"no geomdet for: "<<dd.rawId()<<". will skip.";
      continue;}
    double d=gd->surface().toGlobal(psit->localPosition()).mag();
    if (d>dLim ){
      dLim=d;
      psimhit=&(*psit);
      plane=&gd->surface();}
  }


  if (psimhit && plane){
    //build a trajectorystate on this surface    
    SurfaceSideDefinition::SurfaceSide surfaceside = SurfaceSideDefinition::atCenterOfSurface;
    GlobalPoint initialPoint=plane->toGlobal(psimhit->localPosition());
    GlobalVector initialMomentum=plane->toGlobal(psimhit->momentumAtEntry());
    int initialCharge =  (psimhit->particleType()>0) ? -1:1;
    CartesianTrajectoryError initialCartesianErrors; //no error at initial state  
    const GlobalTrajectoryParameters initialParameters(initialPoint,initialMomentum,initialCharge,thePropagator->magneticField());
    return TrajectoryStateOnSurface(initialParameters,initialCartesianErrors,*plane,surfaceside);}
  else{
    //    edm::LogError("TrackAssociatorByPositionImpl")<<"no corresponding PSimHit for a tracking particle. will fail.";
    return TrajectoryStateOnSurface();}
}

FreeTrajectoryState TrackAssociatorByPositionImpl::getState(const reco::Track & track)const{
  //may be you want to do more than that if track does not go to IP
  return trajectoryStateTransform::initialFreeState(track,thePropagator->magneticField());
}

double TrackAssociatorByPositionImpl::quality(const TrajectoryStateOnSurface & tr, const TrajectoryStateOnSurface & sim) const {
  switch(theMethod){
  case Method::chi2:
    {
      AlgebraicVector5 v(tr.localParameters().vector() - sim.localParameters().vector());
      AlgebraicSymMatrix55 m(tr.localError().matrix());
      int ierr = ! m.Invert();
      if (ierr!=0) edm::LogInfo("TrackAssociatorByPositionImpl")<<"error inverting the error matrix:\n"<<m;
      double est = ROOT::Math::Similarity(v,m);
      return est;
      break;
    }
  case Method::dist:
    {
      return (tr.globalPosition() - sim.globalPosition()).mag();
      break;
    }
  case Method::momdr:
    {
      return  (deltaR<double>(tr.globalDirection().eta(),tr.globalDirection().phi(),
			      sim.globalDirection().eta(),sim.globalDirection().phi()));
      break;
    }
  case Method::posdr:
    {
      return (deltaR<double>(tr.globalPosition().eta(),tr.globalPosition().phi(),
			     sim.globalPosition().eta(),sim.globalPosition().phi()));
      break;
    }
  }
  //should never be reached
  edm::LogError("TrackAssociatorByPositionImpl")<<"option: "<<static_cast<int>(theMethod)<<" has not been recognized. association has no meaning.";
  return -1;
}


RecoToSimCollection TrackAssociatorByPositionImpl::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& tCH, 
                                                                      const edm::RefVector<TrackingParticleCollection>& tPCH) const {
  RecoToSimCollection  outputCollection;
  //for each reco track find a matching tracking particle
  std::pair<unsigned int,unsigned int> minPair;
  const double dQmin_default=1542543;
  double dQmin=dQmin_default;

  //cdj edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;

  //warning: make sure the TP collection used in the map is the same used in the associator!
  //e->getByLabel(_simHitTpMapTag,simHitsTPAssoc);

  for (unsigned int Ti=0; Ti!=tCH.size();++Ti){
    //initial state (initial OR inner OR outter)
    FreeTrajectoryState iState = getState(*(tCH)[Ti]);

    bool atLeastOne=false;
    //    for each tracking particle, find a state position and the plane to propagate the track to.
    for (unsigned int TPi=0;TPi!=tPCH.size();++TPi) {
      //get a state in the muon system 
      TrajectoryStateOnSurface simReferenceState = getState((tPCH)[TPi],*theSimHitsTPAssoc);
      if (!simReferenceState.isValid()) continue;

      //propagate the TRACK to the surface
      TrajectoryStateOnSurface trackReferenceState = thePropagator->propagate(iState,simReferenceState.surface());
      if (!trackReferenceState.isValid()) continue; 
      
      //comparison
      double dQ= quality(trackReferenceState,simReferenceState);
      if (dQ < theQCut){
	atLeastOne=true;
	outputCollection.insert(tCH[Ti],
				std::make_pair(tPCH[TPi],-dQ));//association map with quality, is order greater-first
	edm::LogVerbatim("TrackAssociatorByPositionImpl")<<"track number: "<<Ti
						     <<" associated with dQ: "<<dQ
						     <<" to TrackingParticle number: " <<TPi;}
      if (dQ < dQmin){
	dQmin=dQ;
	minPair = std::make_pair(Ti,TPi);}
    }//loop over tracking particles
    if (theMinIfNoMatch && !atLeastOne && dQmin!=dQmin_default){
      outputCollection.insert(tCH[minPair.first],
			      std::make_pair(tPCH[minPair.second],-dQmin));}
  }//loop over tracks
  outputCollection.post_insert();
  return outputCollection;
}



SimToRecoCollection TrackAssociatorByPositionImpl::associateSimToReco(const edm::RefToBaseVector<reco::Track>& tCH, 
                                                                      const edm::RefVector<TrackingParticleCollection>& tPCH) const {
  SimToRecoCollection  outputCollection;
  //for each tracking particle, find matching tracks.

  std::pair<unsigned int,unsigned int> minPair;
  const double dQmin_default=1542543;
  double dQmin=dQmin_default;

  //edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;

  //warning: make sure the TP collection used in the map is the same used in the associator!
  //e->getByLabel(_simHitTpMapTag,simHitsTPAssoc);

  for (unsigned int TPi=0;TPi!=tPCH.size();++TPi){
    //get a state in the muon system
    TrajectoryStateOnSurface simReferenceState= getState((tPCH)[TPi],*theSimHitsTPAssoc);
      
    if (!simReferenceState.isValid()) continue; 
    bool atLeastOne=false;
    //	propagate every track from any state (initial, inner, outter) to the surface 
    //	and make the position test
    for (unsigned int Ti=0; Ti!=tCH.size();++Ti){
      //initial state
      FreeTrajectoryState iState = getState(*(tCH)[Ti]);
	
      //propagation to surface
      TrajectoryStateOnSurface trackReferenceState = thePropagator->propagate(iState,simReferenceState.surface());
      if (!trackReferenceState.isValid()) continue;
	
      //comparison
      double dQ= quality(trackReferenceState, simReferenceState);
      if (dQ < theQCut){
	atLeastOne=true;
	outputCollection.insert(tPCH[TPi],
				std::make_pair(tCH[Ti],-dQ));//association map with quality, is order greater-first
	edm::LogVerbatim("TrackAssociatorByPositionImpl")<<"TrackingParticle number: "<<TPi
						     <<" associated with dQ: "<<dQ
						     <<" to track number: "<<Ti;}
      if (dQ <dQmin){
	dQmin=dQ;
	minPair = std::make_pair(TPi,Ti);}
    }//loop over tracks
    if (theMinIfNoMatch && !atLeastOne && dQmin!=dQmin_default){
      outputCollection.insert(tPCH[minPair.first],
			      std::make_pair(tCH[minPair.second],-dQmin));}
  }//loop over tracking particles
  
  outputCollection.post_insert();
  return outputCollection;
}
