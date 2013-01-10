#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"


GEMSimSetUp::GEMSimSetUp(const edm::ParameterSet& ps)
{
  averageEfficiency_ = ps.getParameter<double>("averageEfficiency");
  averageNoiseRate_ = ps.getParameter<double>("averageNoiseRate");
  averageShapingTime_ = ps.getParameter<double>("averageShapingTime");
}

void GEMSimSetUp::setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
                        std::vector<float> &vcls)
{
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<float> sum_clsize;

  for(unsigned int n = 0; n < vcls.size(); ++n)
    {
      sum_clsize.push_back(vcls[n]);

      if(counter == row * 20)
	{
	  clsMap_[row] = sum_clsize;
	  row++;
	  sum_clsize.clear();
	}
      counter++;
    }

  setupNoise(vnoise);
}


void GEMSimSetUp::setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
                        std::vector<RPCClusterSize::ClusterSizeItem> &vClusterSize)
{
  int clsCounter = 1;
  std::vector<float> clsVect;

  for(auto cls: vClusterSize)
    {
      clsVect.push_back(cls.clusterSize);
      if( !(clsCounter % 100) && clsCounter != 0 )
	{
	  mapDetClsMap_[cls.dpid] = clsVect;
	  clsVect.clear();
	  clsCounter = 0;
	}
      ++clsCounter;
    }

  setupNoise(vnoise);
}


void GEMSimSetUp::setup()
{
  std::vector<RPCStripNoises::NoiseItem> vnoise;
  // Loop over the detIds
  unsigned int n = 0; 
  for(const auto &det: geometry_->dets())
    {
      if( dynamic_cast< GEMEtaPartition* >( det ) != 0 )
	{
	  GEMEtaPartition* roll = dynamic_cast< GEMEtaPartition* >( det );
	  
	  // construct noise item
	  RPCStripNoises::NoiseItem noise;
	  noise.dpid = roll->id();
	  noise.eff = averageEfficiency_;
	  noise.noise = averageNoiseRate_;
	  noise.time = averageShapingTime_;

	  // add noise item to noise vector
	  vnoise[n] = noise;
	}
      ++n;
    }
  setupNoise(vnoise);
}


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


const std::vector<float>& GEMSimSetUp::getEfficiency(uint32_t id)
{
  auto iter = mapDetIdEff_.find(id);
  if(iter == mapDetIdEff_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getEfficiency() - no efficiency information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != 96)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getEfficiency() - efficiency information in a wrong format for DetId "<<id<< "\n";
  }
  return iter->second;
}


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


const std::map< int, std::vector<float> >& GEMSimSetUp::getClsMap()
{
  if(clsMap_.size() != 5)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getClsMap() - cluster size has the wrong format.\n";
  }
  return clsMap_;
}


const std::vector<float>& GEMSimSetUp::getCls(uint32_t id)
{
  auto iter = mapDetClsMap_.find(id);
  if(iter == mapDetClsMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getCls() - no cluster size information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != 100)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMSimSetUp::getCls() - cluster size information in a wrong format for DetId "<<id<< "\n";
  }
  return iter->second;
}


void GEMSimSetUp::setupNoise(std::vector<RPCStripNoises::NoiseItem> &vnoise)
{
  unsigned int n = 0; 
  uint32_t temp = 0; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for(auto noise: vnoise)
  {
    if(n % 96 == 0)
    {
      if(n > 0 )
      {
        mapDetIdNoise_[temp]= vvnoise;
        mapDetIdEff_[temp] = veff;
        bxmap_[GEMDetId(noise.dpid)] = noise.time;

        veff.clear();
        vvnoise.clear();
        vvnoise.push_back(noise.noise);
        veff.push_back(noise.eff);
      }
      else if (n == 0 )
      {
        vvnoise.push_back(noise.noise);
        veff.push_back(noise.eff);
        bxmap_[GEMDetId(noise.dpid)] = noise.time;
      }
    }
    else if (n == vnoise.size() - 1 )
    {
      temp = noise.dpid;
      vvnoise.push_back(noise.noise);
      veff.push_back(noise.eff);
      mapDetIdNoise_[temp]= vvnoise;
      mapDetIdEff_[temp] = veff;
    }
    else
    {
      temp = noise.dpid;
      vvnoise.push_back(noise.noise);
      veff.push_back(noise.eff);
    }
    n++;
  }
}



