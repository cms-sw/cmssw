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

class TrajectoryStateClosestToBeamLineBuilder;


TrackingParticle::Vector
 CosmicParametersDefinerForTP::momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticle& tp) const{
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

  // cout<<"TrackingParticle pdgId = "<<tp.pdgId()<<endl;
  // cout<<"with tp.vertex(): ("<<tp.vertex().x()<<", "<<tp.vertex().y()<<", "<<tp.vertex().z()<<")"<<endl;
  // cout<<"with tp.momentum(): ("<<tp.momentum().x()<<", "<<tp.momentum().y()<<", "<<tp.momentum().z()<<")"<<endl;
  
  GlobalVector finalGV;
  GlobalPoint finalGP;
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
  double radius(9999);
#endif
  bool found(0);
  TrackingParticle::Vector momentum(0,0,0);
  
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
  const vector<PSimHit> & simHits = tp.trackPSimHit(DetId::Tracker);
  for(vector<PSimHit>::const_iterator it=simHits.begin(); it!=simHits.end(); ++it){
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
#endif

  //cout<<"found = "<<found<<endl;
  // cout<<"Closest Hit Position: ("<<finalGP.x()<<", "<<finalGP.y()<<", "<<finalGP.z()<<")"<<endl;
  //cout<<"Momentum at Closest Hit to BL: ("<<finalGV.x()<<", "<<finalGV.y()<<", "<<finalGV.z()<<")"<<endl;

  if(found) 
    {
      FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tp.charge()),theMF.product());
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

TrackingParticle::Point CosmicParametersDefinerForTP::vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TrackingParticle& tp) const{
  
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
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
  double radius(9999);
#endif
  bool found(0);
  TrackingParticle::Point vertex(0,0,0);

#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
  const vector<PSimHit> & simHits = tp.trackPSimHit(DetId::Tracker);
  for(vector<PSimHit>::const_iterator it=simHits.begin(); it!=simHits.end(); ++it){
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
#endif
  if(found)
    {
      FreeTrajectoryState ftsAtProduction(finalGP,finalGV,TrackCharge(tp.charge()),theMF.product());
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
