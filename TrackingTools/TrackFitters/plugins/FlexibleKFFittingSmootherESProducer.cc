
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"


namespace {

class FlexibleKFFittingSmoother GCC11_FINAL : public TrajectoryFitter {

public:
  /// constructor with predefined fitter and smoother and propagator
  FlexibleKFFittingSmoother(const TrajectoryFitter& standardFitter,
			    const TrajectoryFitter& looperFitter) :
      theStandardFitter(standardFitter.clone()),
      theLooperFitter(looperFitter.clone()) {}

  virtual ~FlexibleKFFittingSmoother(){};

  Trajectory fitOne(const Trajectory& t,fitType type) const{ return fitter(type)->fitOne(t,type);}


  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    const TrajectoryStateOnSurface& firstPredTsos,
		    fitType type) const {return fitter(type)->fitOne(aSeed,hits,firstPredTsos,type); }

  Trajectory fitOne(const TrajectorySeed& aSeed,
		    const RecHitContainer& hits,
		    fitType type) const { return fitter(type)->fitOne(aSeed,hits,type); }

  std::unique_ptr<TrajectoryFitter> clone() const override{
    return std::unique_ptr<TrajectoryFitter>(
        new FlexibleKFFittingSmoother(*theStandardFitter,*theLooperFitter));
        }

 // FIXME a prototype:  final inplementaiton may differ
  virtual void setHitCloner(TkCloner const * hc) {
	theStandardFitter->setHitCloner(hc);
        theLooperFitter->setHitCloner(hc);
  }

 private:

        const TrajectoryFitter* fitter(fitType type) const {
      return (type==standard) ? theStandardFitter.get() : theLooperFitter.get();
    }

    const std::unique_ptr<TrajectoryFitter> theStandardFitter;
    const std::unique_ptr<TrajectoryFitter> theLooperFitter;

  };



class  FlexibleKFFittingSmootherESProducer: public edm::ESProducer{
 public:
  FlexibleKFFittingSmootherESProducer(const edm::ParameterSet & p) 
  {
    std::string myname = p.getParameter<std::string>("ComponentName");
    pset_ = p;
    setWhatProduced(this,myname);
  }
  
  ~FlexibleKFFittingSmootherESProducer() {}
  
  boost::shared_ptr<TrajectoryFitter> 
  produce(const TrajectoryFitterRecord & iRecord){ 
    
    std::string standardFitterName = pset_.getParameter<std::string>("standardFitter");
    std::string looperFitterName = pset_.getParameter<std::string>("looperFitter");
    
    edm::ESHandle<TrajectoryFitter> standardFitter;
    edm::ESHandle<TrajectoryFitter> looperFitter;
    
    iRecord.get(standardFitterName,standardFitter);
    iRecord.get(looperFitterName,looperFitter);
    
    _fitter  = boost::shared_ptr<TrajectoryFitter>(new FlexibleKFFittingSmoother(*standardFitter.product(),
									       *looperFitter.product()   ) );
    return _fitter;
  }
  
  
private:
  boost::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};

}



#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(FlexibleKFFittingSmootherESProducer);
