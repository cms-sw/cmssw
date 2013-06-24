#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include <FWCore/Framework/interface/ESHandle.h>
class TrajectoryStateClosestToBeamLineBuilder;

TrackingParticle::Vector
ParametersDefinerForTP::momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup,
	const Charge charge, const Point & vtx, const LorentzVector& lv) const {
  // to add a new implementation for cosmic. For the moment, it is just as for the base class:

  using namespace edm;

  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  TrackingParticle::Vector momentum(0, 0, 0); 

  FreeTrajectoryState ftsAtProduction(GlobalPoint(vtx.x(),vtx.y(),vtx.z()),
				      GlobalVector(lv.x(),lv.y(),lv.z()),
				      TrackCharge(charge),
				      theMF.product());
        
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
  if(tsAtClosestApproach.isValid()){
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    momentum = TrackingParticle::Vector(p.x(), p.y(), p.z());
  }
  return momentum;
}

TrackingParticle::Point ParametersDefinerForTP::vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup,
	const Charge charge, const Point & vtx, const LorentzVector& lv) const {
  // to add a new implementation for cosmic. For the moment, it is just as for the base class:
  using namespace edm;

  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  TrackingParticle::Point vertex(0, 0, 0);
  
  FreeTrajectoryState ftsAtProduction(GlobalPoint(vtx.x(),vtx.y(),vtx.z()),
				      GlobalVector(lv.x(),lv.y(),lv.z()),
				      TrackCharge(charge),
				      theMF.product());
        
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
  if(tsAtClosestApproach.isValid()){
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    vertex = TrackingParticle::Point(v.x()-bs->x0(),v.y()-bs->y0(),v.z()-bs->z0());
  }
  return vertex;
}


TYPELOOKUP_DATA_REG(ParametersDefinerForTP);
