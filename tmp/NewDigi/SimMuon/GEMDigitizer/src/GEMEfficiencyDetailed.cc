#include "SimMuon/GEMDigitizer/src/GEMEfficiencyDetailed.h"


GEMEfficiencyDetailed::GEMEfficiencyDetailed(const edm::ParameterSet& config)
  : GEMEfficiency(config)
{
  std::cout << ">>> Using efficiency model: GEMEfficiencyDetailed" << std::endl;

  const auto pset(config.getParameter<edm::ParameterSet>("efficiencyModelConfig"));
  averageEfficiency_ = pset.getParameter<double>("averageEfficiency");
}


GEMEfficiencyDetailed::~GEMEfficiencyDetailed()
{
  if (flat_) delete flat_;
}


void 
GEMEfficiencyDetailed::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat_ = new CLHEP::RandFlat(eng);
}


void 
GEMEfficiencyDetailed::setUp(std::vector<GEMStripEfficiency::StripEfficiencyItem> efficiencyVector)
{
  efficiencyMap_.clear();
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
	<< "GEMEfficiencyDetailed::setUp() - number of strips per partition in configuration ("
	<< numberOfStripsPerPartition_ << ") is not the same as in geometry (" << nStrips << ")." << std::endl; 
    }
    const float efficiency(flat_->fire(1.));
    std::vector<float> v(numberOfStripsPerPartition_);
    v.clear();
    for (int i=0; i < numberOfStripsPerPartition_; ++i)
    { 
      v.at(i) = efficiency;
    }
    efficiencyMap_[roll->id().rawId()] = v;  
  }
}


const bool 
GEMEfficiencyDetailed::isGoodDetId(const uint32_t detId)
{
  return efficiencyMap_[detId].at(0) > averageEfficiency_;
}





