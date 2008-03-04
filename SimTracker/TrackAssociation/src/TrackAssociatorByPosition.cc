#include "SimTracker/TrackAssociation/interface/TrackAssociatorByPosition.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace reco;

TrajectoryStateOnSurface TrackAssociatorByPosition::getState(const TrackingParticle & simtrack)const{
  //loop over PSimHits
  const PSimHit * psimhit=0;
  const BoundPlane * plane=0;
  double dLim=0;

  //    look for the further most hit beyond a certain limit
  LogDebug("TrackAssociatorByPosition")<<(int)(simtrack.pSimHit_end()-simtrack.pSimHit_begin())<<" PSimHits.";
  for (std::vector<PSimHit> ::const_iterator psit=simtrack.pSimHit_begin();psit!=simtrack.pSimHit_end();++psit){
    //get the detid
    DetId dd(psit->detUnitId());
    LogDebug("TrackAssociatorByPosition")<<psit-simtrack.pSimHit_begin()
					 <<"] PSimHit on: "<<dd.rawId();
    //get the surface from the global geometry
    const GeomDet * gd=theGeometry->idToDet(dd);
    if (!gd){edm::LogError("TrackAssociatorByPosition")<<"no geomdet for: "<<dd.rawId()<<". will fail.";
      return TrajectoryStateOnSurface();}
    double d=gd->surface().toGlobal(psit->localPosition()).mag();
    if (d>dLim ){
      dLim=d;
      psimhit=&(*psit);
      plane=&gd->surface();}
  }


  if (psimhit && plane){
    //build a trajectorystate on this surface    
    SurfaceSide surfaceside = atCenterOfSurface;
    GlobalPoint initialPoint=plane->toGlobal(psimhit->localPosition());
    GlobalVector initialMomentum=plane->toGlobal(psimhit->momentumAtEntry());
    int initialCharge =  (psimhit->particleType()>0) ? -1:1;
    CartesianTrajectoryError initialCartesianErrors(HepSymMatrix(6,0)); //no error at initial state  
    const GlobalTrajectoryParameters initialParameters(initialPoint,initialMomentum,initialCharge,thePropagator->magneticField());
    return TrajectoryStateOnSurface(initialParameters,initialCartesianErrors,*plane,surfaceside);}
  else{
    //    edm::LogError("TrackAssociatorByPosition")<<"no corresponding PSimHit for a tracking particle. will fail.";
    return TrajectoryStateOnSurface();}
}

FreeTrajectoryState TrackAssociatorByPosition::getState(const reco::Track & track)const{
  static TrajectoryStateTransform transformer;
  //may be you want to do more than that if track does not go to IP
  return transformer. initialFreeState(track,thePropagator->magneticField());
}

double TrackAssociatorByPosition::quality(const TrajectoryStateOnSurface & tr, const TrajectoryStateOnSurface & sim) const {
  switch(theMethod){
  case 0:
    {
      AlgebraicVector5 v(tr.localParameters().vector() - sim.localParameters().vector());
      AlgebraicSymMatrix55 m(tr.localError().matrix());
      int ierr = ! m.Invert();
      if (ierr!=0) edm::LogInfo("TrackAssociatorByPosition")<<"error inverting the error matrix:\n"<<m;
      double est = ROOT::Math::Similarity(v,m);
      return est;
      break;
    }
  case 1:
    {
      return (tr.globalPosition() - sim.globalPosition()).mag();
      break;
    }
  case 2:
    {
      return  (deltaR<double>(tr.globalDirection().eta(),tr.globalDirection().phi(),
			      sim.globalDirection().eta(),sim.globalDirection().phi()));
      break;
    }
  case 3:
    {
      return (deltaR<double>(tr.globalPosition().eta(),tr.globalPosition().phi(),
			     sim.globalPosition().eta(),sim.globalPosition().phi()));
      break;
    }
  }
  //should never be reached
  edm::LogError("TrackAssociatorByPosition")<<"option: "<<theMethod<<" has not been recognized. association has no meaning.";
  return -1;
}


RecoToSimCollection TrackAssociatorByPosition::associateRecoToSim(edm::Handle<reco::TrackCollection>& tCH, 
								  edm::Handle<TrackingParticleCollection>& tPCH,
								  const edm::Event * e ) const{
  RecoToSimCollection  outputCollection;
  //for each reco track find a matching tracking particle
  std::pair<uint,uint> minPair;
  const double dQmin_default=1542543;
  double dQmin=dQmin_default;
  for (uint Ti=0; Ti!=tCH->size();++Ti){
    //initial state (initial OR inner OR outter)
    FreeTrajectoryState iState = getState((*tCH)[Ti]);

    bool atLeastOne=false;
    //    for each tracking particle, find a state position and the plane to propagate the track to.
    for (uint TPi=0;TPi!=tPCH->size();++TPi) {
      //get a state in the muon system 
      TrajectoryStateOnSurface simReferenceState = getState((*tPCH)[TPi]);
      if (!simReferenceState.isValid()) continue;

      //propagate the TRACK to the surface
      TrajectoryStateOnSurface trackReferenceState = thePropagator->propagate(iState,simReferenceState.surface());
      if (!trackReferenceState.isValid()) continue; 
      
      //comparison
      double dQ= quality(trackReferenceState,simReferenceState);
      if (dQ < theQCut){
	atLeastOne=true;
	outputCollection.insert(reco::TrackRef(tCH,Ti),
				std::make_pair(edm::Ref<TrackingParticleCollection>(tPCH,TPi),-dQ));
	edm::LogVerbatim("TrackAssociatorByPosition")<<"track number: "<<Ti
						     <<" associated with dQ: "<<dQ
						     <<" to TrackingParticle number: " <<TPi;}
      if (dQ < dQmin){
	dQmin=dQ;
	minPair = std::make_pair(Ti,TPi);}
    }//loop over tracking particles
    if (theMinIfNoMatch && !atLeastOne && dQmin!=dQmin_default){
      outputCollection.insert(reco::TrackRef(tCH,minPair.first),
			       std::make_pair(edm::Ref<TrackingParticleCollection>(tPCH,minPair.second),-dQmin));}
  }//loop over tracks
  outputCollection.post_insert();
  return outputCollection;
}



SimToRecoCollection TrackAssociatorByPosition::associateSimToReco(edm::Handle<reco::TrackCollection>& tCH, 
							      edm::Handle<TrackingParticleCollection>& tPCH,
							      const edm::Event * e ) const {
  SimToRecoCollection  outputCollection;
  //for each tracking particle, find matching tracks.

  std::pair<uint,uint> minPair;
  const double dQmin_default=1542543;
  double dQmin=dQmin_default;
  for (uint TPi=0;TPi!=tPCH->size();++TPi){
    //get a state in the muon system
    TrajectoryStateOnSurface simReferenceState= getState((*tPCH)[TPi]);
      
    if (!simReferenceState.isValid()) continue; 
    bool atLeastOne=false;
    //	propagate every track from any state (initial, inner, outter) to the surface 
    //	and make the position test
    for (uint Ti=0; Ti!=tCH->size();++Ti){
      //initial state
      FreeTrajectoryState iState = getState((*tCH)[Ti]);
	
      //propagation to surface
      TrajectoryStateOnSurface trackReferenceState = thePropagator->propagate(iState,simReferenceState.surface());
      if (!trackReferenceState.isValid()) continue;
	
      //comparison
      double dQ= quality(trackReferenceState, simReferenceState);
      if (dQ < theQCut){
	atLeastOne=true;
	outputCollection.insert(edm::Ref<TrackingParticleCollection>(tPCH,TPi),
				std::make_pair(reco::TrackRef(tCH,Ti),-dQ));
	edm::LogVerbatim("TrackAssociatorByPosition")<<"TrackingParticle number: "<<TPi
						     <<" associated with dQ: "<<dQ
						     <<" to track number: "<<Ti;}
      if (dQ <dQmin){
	dQmin=dQ;
	minPair = std::make_pair(TPi,Ti);}
    }//loop over tracks
    if (theMinIfNoMatch && !atLeastOne && dQmin!=dQmin_default){
      outputCollection.insert(edm::Ref<TrackingParticleCollection>(tPCH,minPair.first),
			      std::make_pair(reco::TrackRef(tCH,minPair.second),-dQmin));}
  }//loop over tracking particles
  
  outputCollection.post_insert();
  return outputCollection;
}
