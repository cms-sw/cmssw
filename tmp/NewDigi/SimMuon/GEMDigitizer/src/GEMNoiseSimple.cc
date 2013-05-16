#include "SimMuon/GEMDigitizer/src/GEMNoiseSimple.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <iostream>


GEMNoiseSimple::GEMNoiseSimple(const edm::ParameterSet& config)
  : GEMNoise(config)
{
  std::cout << ">>> Using noise model: GEMNoiseSimple" << std::endl;
  
  const auto timingSet(config.getParameter<edm::ParameterSet>("timingModelConfig"));
  bxWidth_ = timingSet.getParameter<double>("bxWidth");
  maxBunch_= timingSet.getParameter<double>("maxBunch");
  minBunch_= timingSet.getParameter<double>("minBunch");
  const auto noiseSet(config.getParameter<edm::ParameterSet>("noiseModelConfig"));
  averageNoiseRate_ = noiseSet.getParameter<double>("averageNoiseRate");
}


GEMNoiseSimple::~GEMNoiseSimple()
{
  if (flat1_) delete flat1_;
  if (flat2_) delete flat2_;
  if (poisson_) delete poisson_;
}


void
GEMNoiseSimple::setRandomEngine(CLHEP::HepRandomEngine& eng) 
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  poisson_ = new CLHEP::RandPoissonQ(eng);
}


void 
GEMNoiseSimple::setUp(std::vector<GEMStripNoise::StripNoiseItem> vNoise)
{
  noiseRateMap_.clear();
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
	<< "GEMNoiseSimple::setUp() - number of strips per partition in configuration ("
	<< numberOfStripsPerPartition_ << ") is not the same as in geometry (" << nStrips << ")." << std::endl; 
    }
    const float noise(averageNoiseRate_);
    std::vector<float> v(numberOfStripsPerPartition_);
    v.clear();
    for (int i=0; i < numberOfStripsPerPartition_; ++i)
    { 
      v.at(i) = noise;
    }
    noiseRateMap_[roll->id().rawId()] = v;  
  }
}


const std::set<std::pair<int, int> >
GEMNoiseSimple::simulateNoise(const GEMEtaPartition* roll) 
{
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float striplength((top_->stripLength()));
  const int nStrips(roll->nstrips());
  const float xmax((top_->localPosition((float) nStrips)).x());
  const float xmin((top_->localPosition(0.)).x());
  const double area(striplength*(xmax-xmin));
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const double averageNoise(noiseRateMap_[roll->id().rawId()].at(0) * nBxing * bxWidth_ * area * 1.0e-9);
  const int nHits(poisson_->fire(averageNoise));

  std::set<std::pair<int, int> > output;
  for (int i(0); i< nHits; ++i){
    const int strip(static_cast<int>(flat1_->fire(1,nStrips)));
    const int timeHit(static_cast<int>(flat2_->fire(nBxing)) + minBunch_);
    std::pair<int, int> digi(strip,timeHit);
    output.insert(digi);
  }
  return output;
}
