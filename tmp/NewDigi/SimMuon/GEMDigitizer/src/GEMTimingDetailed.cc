#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "SimMuon/GEMDigitizer/src/GEMTimingDetailed.h"
#include "CLHEP/Random/RandGaussQ.h"


GEMTimingDetailed::GEMTimingDetailed(const edm::ParameterSet& config)
  : GEMTiming(config)
  , gauss1_(0)
  , gauss2_(0)
 {
   std::cout << ">>> Using timing model: GEMTimingDetailed" << std::endl;

  const auto pset(config.getParameter<edm::ParameterSet>("timingModelConfig"));
  timeCalibrationOffset_ = pset.getParameter<double>("timeCalibrationOffset");
  timeResolution_ = pset.getParameter<double>("timeResolution");
  averageShapingTime_ = pset.getParameter<double>("averageShapingTime");
  timeJitter_ = pset.getParameter<double>("timeJitter");
  signalPropagationSpeed_ = pset.getParameter<double>("signalPropagationSpeed");
  cosmics_ = pset.getParameter<bool>("cosmics");
  bxWidth_ = pset.getParameter<double>("bxWidth");
  numberOfStripsPerPartition_ = config.getParameter<int>("numberOfStripsPerPartition");
  
  // signal propagation speed in vacuum in [m/s]
  const double lightSpeed(299792458);
  // signal propagation speed in material in [cm/ns]
  signalPropagationSpeed_ = signalPropagationSpeed_ * lightSpeed * 1e+2 * 1e-9;
}


GEMTimingDetailed::~GEMTimingDetailed()
{
  if (gauss1_) delete gauss1_;
  if (gauss2_) delete gauss2_;
}

void 
GEMTimingDetailed::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
}


void 
GEMTimingDetailed::setUp(std::vector<GEMStripTiming::StripTimingItem> timingVector)
{
  calibrationTimeMap_.clear();
  // Loop over the detIds                                                                                                                                             
  for(const auto &det: getGeometry()->dets())
  {
    const GEMEtaPartition* roll(dynamic_cast<GEMEtaPartition*>(det));
    
    // check for valid rolls     
    if(!roll) continue;
    const int nStrips(roll->nstrips());
    if (numberOfStripsPerPartition_ != nStrips)
    {
      throw cms::Exception("DataCorrupt") 
	<< "GEMTimingDetailed::setUp() - number of strips per partition in configuration ("
	<< numberOfStripsPerPartition_ << ") is not the same as in geometry (" << nStrips << ")." << std::endl; 
    }

    std::vector<float> v(numberOfStripsPerPartition_);
    v.clear();
    for (int i=0; i < numberOfStripsPerPartition_; ++i)
    { 
      v.at(i) = timeCalibrationOffset_;
    }
    calibrationTimeMap_[roll->id().rawId()] = v;  
  }
}


const int 
GEMTimingDetailed::getSimHitBx(const PSimHit* simhit)
{
  const GEMGeometry* geometry(getGeometry());
  const GEMDetId detId(simhit->detUnitId());
  const GeomDetUnit* geomDetUnit(geometry->idToDetUnit(detId));
  const GEMEtaPartition* roll(dynamic_cast<const GEMEtaPartition*>(geomDetUnit));

  int bx(-999);
  // check for valid eta partition
  if (!roll) return bx;
  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry")
      << "GEMTimingDetailed::getSimHitBx() - this GEM id is from barrel, which cannot happen:  " << roll->id() << "\n";
  }
 
  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const LocalPoint simHitPos(simhit->localPosition());
  const float distanceFromEdge(halfStripLength - simHitPos.y());
  
  // time of flight
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  const float randomJitterTime(gauss1_->fire(0., timeJitter_));
  // random Gaussian time correction due to the finite timing resolution of the detector
  const float randomResolutionTime(gauss2_->fire(0., timeResolution_));
  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge/signalPropagationSpeed_);
  // calibration time
  const float calibrationTime(calibrationTimeMap_[simhit->detUnitId()].at(0));

  const float simhitTime(tof + (averageShapingTime_ + randomResolutionTime) + (averagePropagationTime + randomJitterTime));
  const float referenceTime(calibrationTime + halfStripLength/signalPropagationSpeed_ + averageShapingTime_);
  const float timeDifference(cosmics_ ? (simhitTime - referenceTime)/COSMIC_PAR : simhitTime - referenceTime);

  // assign the bunch crossing
  bx = static_cast<int>(std::round(timeDifference/bxWidth_));
  
  // check time
  const bool debug(false);
  if (debug)
  {
    std::cout<<"checktime "<<bx<<" "<<timeDifference<<" "<<simhitTime<<" "<<referenceTime<<" "<<tof<<" "<<averagePropagationTime<<std::endl;
  }
  return bx;
}

