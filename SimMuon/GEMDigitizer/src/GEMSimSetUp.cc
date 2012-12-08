#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

using namespace std;


void GEMSimSetUp::setupNoise(vector<RPCStripNoises::NoiseItem> &vnoise)
{
  unsigned int n = 0; 
  uint32_t temp = 0; 
  vector<float> veff, vvnoise;
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


void GEMSimSetUp::setup(vector<RPCStripNoises::NoiseItem> &vnoise,
                        vector<float> &vcls)
{
  unsigned int counter = 1;
  unsigned int row = 1;
  vector<float> sum_clsize;

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


void GEMSimSetUp::setup(vector<RPCStripNoises::NoiseItem> &vnoise,
                        vector<RPCClusterSize::ClusterSizeItem> &vClusterSize)
{
  int clsCounter = 1;
  vector<float> clsVect;

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


const vector<float>& GEMSimSetUp::getNoise(uint32_t id)
{
  auto iter = mapDetIdNoise_.find(id);
  if(iter == mapDetIdNoise_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - no noise information for DetId "<<id<< "\n";
  }
  return iter->second;
}


const vector<float>& GEMSimSetUp::getEff(uint32_t id)
{
  auto iter = mapDetIdEff_.find(id);
  if(iter == mapDetIdEff_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - no efficiency information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != 96)
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - efficiency information in a wrong format for DetId "<<id<< "\n";
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
      << "Exception coming from GEMSimSetUp - no timing information for rpcid.rawId() "<<detid.rawId()<< "\n";
  }
  return iter->second;
}


const map< int, vector<float> >& GEMSimSetUp::getClsMap()
{
  if(clsMap_.size() != 5)
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - cluster size - wrong format.\n";
  }
  return clsMap_;
}


const vector<float>& GEMSimSetUp::getCls(uint32_t id)
{
  auto iter = mapDetClsMap_.find(id);
  if(iter == mapDetClsMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - no cluster size information for DetId "<<id<< "\n";
  }
  if((iter->second).size() != 100)
  {
    throw cms::Exception("DataCorrupt") 
      << "Exception coming from GEMSimSetUp - cluster size information in a wrong format for DetId "<<id<< "\n";
  }
  return iter->second;
}
