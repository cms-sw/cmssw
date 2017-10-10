


// to be included only here...
#include "KFFittingSmoother.h"



#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"



namespace {

  class  KFFittingSmootherESProducer final: public edm::ESProducer{
  public:
    KFFittingSmootherESProducer(const edm::ParameterSet & p) 
    {
      std::string myname = p.getParameter<std::string>("ComponentName");
      pset_ = p;
      setWhatProduced(this,myname);
    }
    
    
    static void  fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName","KFFittingSmoother");
      desc.add<std::string>("Fitter","KFFitter");
      desc.add<std::string>("Smoother","KFSmoother");
      KFFittingSmoother::fillDescriptions(desc);
      descriptions.add("KFFittingSmoother", desc);
    }
    
    
    ~KFFittingSmootherESProducer() override {}
    
    std::shared_ptr<TrajectoryFitter> 
    produce(const TrajectoryFitterRecord & iRecord){ 
      
      
      edm::ESHandle<TrajectoryFitter> fit;
      edm::ESHandle<TrajectorySmoother> smooth;

      iRecord.get(pset_.getParameter<std::string>("Fitter"), fit);
      iRecord.get(pset_.getParameter<std::string>("Smoother"), smooth);
      
     return std::shared_ptr<TrajectoryFitter>(new KFFittingSmoother(*fit.product(), *smooth.product(),pset_));
    }

  private:
    edm::ParameterSet pset_;
  };
}

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFFittingSmootherESProducer);
