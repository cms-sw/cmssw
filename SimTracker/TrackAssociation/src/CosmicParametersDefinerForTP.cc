#include "SimTracker/TrackAssociation/interface/CosmicParametersDefinerForTP.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <DataFormats/GeometrySurface/interface/Surface.h>
#include <DataFormats/GeometrySurface/interface/GloballyPositioned.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrajectoryStateClosestToBeamLineBuilder;


TrackingParticle::Vector
 CosmicParametersDefinerForTP::momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticleRef tpr) const{
  // to add a new implementation for cosmic. For the moment, it is just as for the base class:
  using namespace edm;
  using namespace std;
  using namespace reco;

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  // cout<<"TrackingParticle pdgId = "<<tpr->pdgId()<<endl;
  // cout<<"with tpr->vertex(): ("<<tpr->vertex().x()<<", "<<tpr->vertex().y()<<", "<<tpr->vertex().z()<<")"<<endl;
  // cout<<"with tpr->momentum(): ("<<tpr->momentum().x()<<", "<<tpr->momentum().y()<<", "<<tpr->momentum().z()<<")"<<endl;
  
  GlobalVector finalGV;
  GlobalPoint finalGP;
  double radius(9999);
  bool found(0);
  TrackingParticle::Vector momentum(0,0,0);

  if (simHitsTPAssoc.isValid()==0) {
    LogError("TrackAssociation") << "Invalid handle!";
    return momentum;
  }
  std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater 
                                                                                                 // sorting only the cluster is needed
  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(), 
				clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
  for(auto ip = range.first; ip != range.second; ++ip) {
    TrackPSimHitRef it = ip->second;
    const GeomDet* tmpDet  = tracker->idToDet( DetId(it->detUnitId()) ) ;
    LocalVector  lv = it->momentumAtEntry();
    Local3DPoint lp = it->localPosition ();
    GlobalVector gv = tmpDet->surface().toGlobal( lv );
    GlobalPoint  gp = tmpDet->surface().toGlobal( lp );
    if(gp.perp()<radius){
      found=true;
      radius = gp.perp();
      finalGV = gv;
      finalGP = gp;
    }
  }

  //cout<<"found = "<<found<<endl;
  // cout<<"Closest Hit Position: ("<<finalGP.x()<<", "<<finalGP.y()<<", "<<finalGP.z()<<")"<<endl;
  //cout<<"Momentum at Closest Hit to BL: ("<<finalGV.x()<<", "<<finalGV.y()<<", "<<finalGV.z()<<")"<<endl;

  if(found) 
    {
      FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tpr->charge()),theMF.product());
      TSCBLBuilderNoMaterial tscblBuilder;
      TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
      if(tsAtClosestApproach.isValid()){
	GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
	momentum = TrackingParticle::Vector(p.x(), p.y(), p.z());
      }
      return momentum;
    }
  return momentum;
}

TrackingParticle::Point CosmicParametersDefinerForTP::vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticleRef tpr) const{
  
  using namespace edm;
  using namespace std;
  using namespace reco;

  ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  GlobalVector finalGV;
  GlobalPoint finalGP;
  double radius(9999);
  bool found(0);
  TrackingParticle::Point vertex(0,0,0);

  if (simHitsTPAssoc.isValid()==0) {
    LogError("TrackAssociation") << "Invalid handle!";
    return vertex;
  }
  std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(tpr,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater 
                                                                                                 // sorting only the cluster is needed
  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(), 
				clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
  for(auto ip = range.first; ip != range.second; ++ip) {
    TrackPSimHitRef it = ip->second;
    const GeomDet* tmpDet  = tracker->idToDet( DetId(it->detUnitId()) ) ;
    LocalVector  lv = it->momentumAtEntry();
    Local3DPoint lp = it->localPosition ();
    GlobalVector gv = tmpDet->surface().toGlobal( lv );
    GlobalPoint  gp = tmpDet->surface().toGlobal( lp );
    if(gp.perp()<radius){
      found=true;
      radius = gp.perp();
      finalGV = gv;
      finalGP = gp;
    }
  }
  if(found)
    {
      FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tpr->charge()),theMF.product());
      TSCBLBuilderNoMaterial tscblBuilder;
      TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
      if(tsAtClosestApproach.isValid()){
	GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
	vertex = TrackingParticle::Point(v.x()-bs->x0(),v.y()-bs->y0(),v.z()-bs->z0());
      }
      return vertex;
    }
  return vertex;
}


TYPELOOKUP_DATA_REG(CosmicParametersDefinerForTP);
