#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include <memory>

namespace {

  class  KFTrajectoryFitterESProducer: public edm::ESProducer{
  public:
    KFTrajectoryFitterESProducer(const edm::ParameterSet & p);
    ~KFTrajectoryFitterESProducer() override; 
    std::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
    
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName","KFFitter");
      desc.add<std::string>("Propagator","PropagatorWithMaterial");
      desc.add<std::string>("Updator","KFUpdator");
      desc.add<std::string>("Estimator","Chi2");
      desc.add<std::string>("RecoGeometry","GlobalDetLayerGeometry");
      desc.add<int>("minHits",3);
      descriptions.add("KFTrajectoryFitter", desc);
    }

    
  private:
    edm::ParameterSet pset_;
  };

  KFTrajectoryFitterESProducer::KFTrajectoryFitterESProducer(const edm::ParameterSet & p) 
  {
    std::string myname = p.getParameter<std::string>("ComponentName");
    pset_ = p;
    setWhatProduced(this,myname);
  }
  
  KFTrajectoryFitterESProducer::~KFTrajectoryFitterESProducer() {}
  
  std::shared_ptr<TrajectoryFitter>
  KFTrajectoryFitterESProducer::produce(const TrajectoryFitterRecord & iRecord){ 
    
    std::string pname = pset_.getParameter<std::string>("Propagator");
    std::string uname = pset_.getParameter<std::string>("Updator");
    std::string ename = pset_.getParameter<std::string>("Estimator");
    std::string gname = pset_.getParameter<std::string>("RecoGeometry");
    int minHits = pset_.getParameter<int>("minHits");
    
    edm::ESHandle<Propagator> prop;
    edm::ESHandle<TrajectoryStateUpdator> upd;
    edm::ESHandle<Chi2MeasurementEstimatorBase> est;
    edm::ESHandle<DetLayerGeometry> geo;
    
    iRecord.getRecord<TrackingComponentsRecord>().get(pname, prop);
    iRecord.getRecord<TrackingComponentsRecord>().get(uname, upd);
    iRecord.getRecord<TrackingComponentsRecord>().get(ename, est);
    iRecord.getRecord<RecoGeometryRecord>().get(gname,geo);

    return std::make_shared<KFTrajectoryFitter>(prop.product(),
				                upd.product(),
				                est.product(),
				                minHits,
				                geo.product());
  }

}

#include "FWCore/Framework/interface/ModuleFactory.h"


DEFINE_FWK_EVENTSETUP_MODULE(KFTrajectoryFitterESProducer);
