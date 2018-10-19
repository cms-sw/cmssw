
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"


namespace {

  class FlexibleKFFittingSmoother final : public TrajectoryFitter {
  public:
        ~FlexibleKFFittingSmoother() override{}

  private:
    /// constructor with predefined fitter and smoother and propagator
    FlexibleKFFittingSmoother(const TrajectoryFitter& standardFitter,
			      const TrajectoryFitter& looperFitter) :
      theStandardFitter(standardFitter.clone()),
      theLooperFitter(looperFitter.clone()) {}
    
    
    Trajectory fitOne(const Trajectory& t,fitType type) const override { return fitter(type)->fitOne(t,type);}
    
    
    Trajectory fitOne(const TrajectorySeed& aSeed,
		      const RecHitContainer& hits,
		      const TrajectoryStateOnSurface& firstPredTsos,
		      fitType type) const override {return fitter(type)->fitOne(aSeed,hits,firstPredTsos,type); }
    
    Trajectory fitOne(const TrajectorySeed& aSeed,
		      const RecHitContainer& hits,
		      fitType type) const override  { return fitter(type)->fitOne(aSeed,hits,type); }
    
    std::unique_ptr<TrajectoryFitter> clone() const override {
      return std::unique_ptr<TrajectoryFitter>(new FlexibleKFFittingSmoother(*theStandardFitter,*theLooperFitter));
    }
    
    // FIXME a prototype:  final inplementaiton may differ
    void setHitCloner(TkCloner const * hc) override {
      theStandardFitter->setHitCloner(hc);
      theLooperFitter->setHitCloner(hc);
    }
    
  private:
    
    const TrajectoryFitter* fitter(fitType type) const {
      return (type==standard) ? theStandardFitter.get() : theLooperFitter.get();
    }
    
    const std::unique_ptr<TrajectoryFitter> theStandardFitter;
    const std::unique_ptr<TrajectoryFitter> theLooperFitter;


    friend class  FlexibleKFFittingSmootherESProducer;
    
  };
  
  
  
  class  FlexibleKFFittingSmootherESProducer: public edm::ESProducer{
  public:
    FlexibleKFFittingSmootherESProducer(const edm::ParameterSet & p) 
    {
      std::string myname = p.getParameter<std::string>("ComponentName");
      pset_ = p;
      setWhatProduced(this,myname);
    }
    
    ~FlexibleKFFittingSmootherESProducer() override {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("ComponentName","FlexibleKFFittingSmoother");
      desc.add<std::string>("standardFitter","KFFittingSmootherWithOutliersRejectionAndRK");
      desc.add<std::string>("looperFitter","LooperFittingSmoother");
      descriptions.add("FlexibleKFFittingSmoother", desc);
    }
    
    std::unique_ptr<TrajectoryFitter> 
    produce(const TrajectoryFitterRecord & iRecord){ 
      
      edm::ESHandle<TrajectoryFitter> standardFitter;
      edm::ESHandle<TrajectoryFitter> looperFitter;
      
      iRecord.get(pset_.getParameter<std::string>("standardFitter"),standardFitter);
      iRecord.get(pset_.getParameter<std::string>("looperFitter"),looperFitter);
      
      return std::unique_ptr<TrajectoryFitter>(new FlexibleKFFittingSmoother(*standardFitter.product(),
                                                                             *looperFitter.product()));
    }
    
    
  private:
    edm::ParameterSet pset_;
  };
  
}



#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(FlexibleKFFittingSmootherESProducer);
