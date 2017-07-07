#include "RecoTracker/TransientTrackingRecHit/plugins/TkTransientTrackingRecHitBuilderESProducer.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

TkTransientTrackingRecHitBuilderESProducer::TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

TkTransientTrackingRecHitBuilderESProducer::~TkTransientTrackingRecHitBuilderESProducer() {}

std::shared_ptr<TransientTrackingRecHitBuilder> 
TkTransientTrackingRecHitBuilderESProducer::produce(const TransientRecHitRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }

  std::string sname = pset_.getParameter<std::string>("StripCPE");
  std::string pname = pset_.getParameter<std::string>("PixelCPE");
  std::string mname = pset_.getParameter<std::string>("Matcher");

  
  edm::ESHandle<StripClusterParameterEstimator> se; 
  edm::ESHandle<PixelClusterParameterEstimator> pe; 
  edm::ESHandle<SiStripRecHitMatcher>           me;
  const StripClusterParameterEstimator * sp ;
  const PixelClusterParameterEstimator * pp ;
  const SiStripRecHitMatcher           * mp ;
    
  if (sname == "Fake") {
    sp = nullptr;
  }else{
    iRecord.getRecord<TkStripCPERecord>().get( sname, se );     
    sp = se.product();
  }
  
  if (pname == "Fake") {
    pp = nullptr;
  }else{
    iRecord.getRecord<TkPixelCPERecord>().get( pname, pe );     
    pp = pe.product();
  }
  
  if (mname == "Fake") {
    mp = nullptr;
  }else{
    iRecord.getRecord<TkStripCPERecord>().get( mname, me );     
    mp = me.product();
  }

  bool computeCoarseLocalPositionFromDisk = pset_.getParameter<bool>("ComputeCoarseLocalPositionFromDisk");
  if (computeCoarseLocalPositionFromDisk)
    edm::LogWarning("TkTransientTrackingRecHitBuilderESProducer")<<" The tracking rec hit positions and errors are not a persistent in data formats.\n"
								 <<" They are not available from disk.\n"
								 <<" However, TkTransientTrackingRecHitBuilderESProducer::ComputeCoarseLocalPositionFromDisk=True \n"
								 <<" will make the coarse estimation of this position/error available without track refit.\n"
								 <<" Position/error obtained from rechit with already defined position/error are not recomputed.\n"
								 <<" Position/error obtained from track refit are precise.";
  
  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );     
  
  //For Phase2 upgrade
  std::string p2OTname = "";
  if(pset_.existsAs<std::string>("Phase2StripCPE")){
    p2OTname = pset_.getParameter<std::string>("Phase2StripCPE");
  }
  edm::ESHandle<ClusterParameterEstimator<Phase2TrackerCluster1D> > p2OTe;
  const ClusterParameterEstimator<Phase2TrackerCluster1D> * p2OTp;

  if (p2OTname != "") {
    iRecord.getRecord<TkStripCPERecord>().get( p2OTname, p2OTe );
    p2OTp = p2OTe.product();
    _builder  = std::make_shared<TkTransientTrackingRecHitBuilder>(pDD.product(), pp, p2OTp);
  } else {
    _builder  = std::make_shared<TkTransientTrackingRecHitBuilder>(pDD.product(), pp, sp, mp, computeCoarseLocalPositionFromDisk);
  }

  return _builder;
}


