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

ParticleBase::Vector
ParametersDefinerForTP::momentum(const edm::Event& iEvent, const edm::EventSetup& iSetup, const ParticleBase& tp) const{
  // to add a new implementation for cosmic. For the moment, it is just as for the base class:

  using namespace edm;

  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  ParticleBase::Vector momentum(0, 0, 0); 

  FreeTrajectoryState ftsAtProduction(GlobalPoint(tp.vertex().x(),tp.vertex().y(),tp.vertex().z()),
				      GlobalVector(tp.momentum().x(),tp.momentum().y(),tp.momentum().z()),
				      TrackCharge(tp.charge()),
				      theMF.product());
        
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
  if(tsAtClosestApproach.isValid()){
    GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
    momentum = ParticleBase::Vector(p.x(), p.y(), p.z());
  }
  return momentum;
}

ParticleBase::Point ParametersDefinerForTP::vertex(const edm::Event& iEvent, const edm::EventSetup& iSetup, const ParticleBase& tp) const{
  // to add a new implementation for cosmic. For the moment, it is just as for the base class:
  using namespace edm;

  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  
  edm::Handle<reco::BeamSpot> bs;
  iEvent.getByLabel(InputTag("offlineBeamSpot"),bs);

  ParticleBase::Point vertex(0, 0, 0);
  
  FreeTrajectoryState ftsAtProduction(GlobalPoint(tp.vertex().x(),tp.vertex().y(),tp.vertex().z()),
				      GlobalVector(tp.momentum().x(),tp.momentum().y(),tp.momentum().z()),
				      TrackCharge(tp.charge()),
				      theMF.product());
        
  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,*bs);//as in TrackProducerAlgorithm
  if(tsAtClosestApproach.isValid()){
    GlobalPoint v = tsAtClosestApproach.trackStateAtPCA().position();
    vertex = ParticleBase::Point(v.x()-bs->x0(),v.y()-bs->y0(),v.z()-bs->z0());
  }
  return vertex;
}


TYPELOOKUP_DATA_REG(ParametersDefinerForTP);
