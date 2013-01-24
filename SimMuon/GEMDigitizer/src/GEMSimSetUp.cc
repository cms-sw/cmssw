#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GEMSimSetUp::GEMSimSetUp(const edm::ParameterSet& config)
{
  auto pset = config.getParameter<edm::ParameterSet>("digiModelConfig");

  averageEfficiency_ = pset.getParameter<double>("averageEfficiency");
  averageNoiseRate_ = pset.getParameter<double>("averageNoiseRate");
  timeCalibrationOffset_ = pset.getParameter<double>("timeCalibrationOffset");
  numberOfStripsPerPartition_ = pset.getParameter<int>("numberOfStripsPerPartition");
}

void GEMSimSetUp::setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
                        std::vector<float> &vcluster)
{
  std::cout << "GEMSimSetUp::setup()" << std::endl;
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<float> sum_clsize;

  for(unsigned int n = 0; n < vcluster.size(); ++n)
    {
      sum_clsize.push_back(vcluster[n]);

      if(counter == row * 20)
	{
	  clusterMap_[row] = sum_clsize;
	  row++;
	  sum_clsize.clear();
	}
      counter++;
    }

  setupNoiseAndEfficiency(vnoise);
}


void GEMSimSetUp::setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
                        std::vector<RPCClusterSize::ClusterSizeItem> &vClusterSize)
{
  int clusterCounter = 1;
  std::vector<float> clusterVect;

  for(auto cluster: vClusterSize)
    {
      clusterVect.push_back(cluster.clusterSize);
      if( !(clusterCounter % 100) && clusterCounter != 0 )
	{
	  mapDetClusterMap_[cluster.dpid] = clusterVect;
	  clusterVect.clear();
	  clusterCounter = 0;
	}
      ++clusterCounter;
    }

  setupNoiseAndEfficiency(vnoise);
}


// setup the noise vector
void GEMSimSetUp::setup()
{
  /*
    Container for the noise items. Its total size is 331776
    - 2 stations
    - 2 layers/station
    - 36 chambers/layer
    - 6 eta partitions/chamber
    - 384 strips/eta partition
  */ 
  std::vector<RPCStripNoises::NoiseItem> vnoise;
  vnoise.reserve(geometry_->dets().size() * numberOfStripsPerPartition_);

  // Loop over the detIds                                                                                                                                             
  for(const auto &det: geometry_->dets())
    {
      GEMEtaPartition* roll = dynamic_cast< GEMEtaPartition* >( det );
      
      // check for valid rolls     
      if(roll == nullptr) continue;

      const int nStrips = roll->nstrips();
      if (numberOfStripsPerPartition_ != nStrips)
	{
	  throw cms::Exception("DataCorrupt") 
	    << "GEMSimSetUp::setup() - numberOfStripsPerPartition given in configuration "
	    <<numberOfStripsPerPartition_ <<" is not the same as in geometry "<<nStrips;
	}
      
      // Loop over the strips                                                                                                                                          
      for(int iStrip=0; iStrip <= nStrips-1; ++iStrip)
        {
          // construct noise item for each strip
	  RPCStripNoises::NoiseItem noise;
          noise.dpid = roll->id();
          noise.eff = averageEfficiency_;
          noise.noise = averageNoiseRate_;
          noise.time = timeCalibrationOffset_;
	  
          // add noise item to noise vector                                                                                                                         
          vnoise.push_back(noise);
	}
      setupTimeCalibration(det->geographicalId(),timeCalibrationOffset_);
    }
  setupNoiseAndEfficiency(vnoise);
}

// return the vector of noise values for a particular chamber
const std::vector<float>& GEMSimSetUp::getNoise(uint32_t id)
{
  auto iter = mapDetIdNoise_.find(id);
  if(iter == mapDetIdNoise_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getNoise() - no noise information for DetId "<<id<< "\n";
  }
  return iter->second;
}

// return the vector of efficiency values for a particular chamber
const std::vector<float>& GEMSimSetUp::getEfficiency(uint32_t id)
{
  auto iter = mapDetIdEfficiency_.find(id);
  if(iter == mapDetIdEfficiency_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getEfficiency() - no efficiency information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != (unsigned) numberOfStripsPerPartition_)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getEfficiency() - efficiency information in a wrong format for DetId "<<id<< "\n";
  }
  return iter->second;
}

// return the time calibration for this chamber
float GEMSimSetUp::getTime(uint32_t id)
{
  GEMDetId detid(id);
  auto iter = bxmap_.find(detid);
  if(iter == bxmap_.end())
    {
      throw cms::Exception("DataCorrupt") 
	<< "GEMSimSetUp::getTime() - no timing information for gemid.rawId() "<<detid.rawId()<< "\n";
    }
  return iter->second;
}


const std::map< int, std::vector<float> >& GEMSimSetUp::getClusterMap()
{
  if(clusterMap_.size() != 5)
    {
      throw cms::Exception("DataCorrupt") 
	<< "GEMSimSetUp::getClusterMap() - cluster size has the wrong format.\n";
    }
  return clusterMap_;
}


const std::vector<float>& GEMSimSetUp::getCluster(uint32_t id)
{
  auto iter = mapDetClusterMap_.find(id);
  if(iter == mapDetClusterMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getCluster() - no cluster size information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != 100)
  {
   throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getCluster() - cluster size information in a wrong format for DetId "<<id<< "\n";
  }
  return iter->second;
}


void GEMSimSetUp::setupNoiseAndEfficiency(std::vector<RPCStripNoises::NoiseItem> &vnoise)
{
  unsigned int iStrip = 0; 
  uint32_t roll = 0; 
  std::vector<float> vEfficiency, vNoise;
  vEfficiency.clear();
  vNoise.clear();

  // loop over the noise vector
  for(auto noise: vnoise)
    {
      if(iStrip % numberOfStripsPerPartition_ == 0)
	{
	  // the first strip of new chamber
	  if(iStrip > 0)
	    {
	      // fill map with noise and efficiency vectors of the previous chamber
	      mapDetIdNoise_[roll] = vNoise;
	      mapDetIdEfficiency_[roll] = vEfficiency;
	      // clear the vectors and start over
	      vEfficiency.clear();
	      vNoise.clear();
	      vNoise.push_back(noise.noise);
	      vEfficiency.push_back(noise.eff);
	    }
	  // the very first strip in the collection
	  else if (iStrip == 0 )
	    {
	      // nothing to add to map yet
	      vNoise.push_back(noise.noise);
	      vEfficiency.push_back(noise.eff);
	    }
	}
      // the very last strip in the collection
      else if (iStrip == vnoise.size() - 1 )
	{
	  roll = noise.dpid;
	  vNoise.push_back(noise.noise);
	  vEfficiency.push_back(noise.eff);
	  // fill map with noise and efficiency vectors of the last chamber
	  mapDetIdNoise_[roll]= vNoise;
	  mapDetIdEfficiency_[roll] = vEfficiency;
	}
      // a regular strip
      else
	{
	  roll = noise.dpid;
	  vNoise.push_back(noise.noise);
	  vEfficiency.push_back(noise.eff);
	}
      iStrip++;
    }
}


// set up the time calibration for each detId
void GEMSimSetUp::setupTimeCalibration(GEMDetId id, float timing)
{
  bxmap_[id] = timing;      
}




