#include "DataFormats/Common/interface/Handle.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <cmath>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>
#include <utility>
#include <map>

using namespace std;

RPCSimSetUp::RPCSimSetUp(const edm::ParameterSet& ps) {
  _mapDetIdNoise.clear();
  _mapDetIdEff.clear();
  _bxmap.clear();
  _clsMap.clear();
}

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<float>& vcls) {
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  for (unsigned int n = 0; n < vcls.size(); ++n) {
    sum_clsize.push_back(vcls[n]);

    if (counter == row * 20) {
      _clsMap[row] = sum_clsize;
      row++;
      sum_clsize.clear();
    }
    counter++;
  }

  unsigned int n = 0;
  uint32_t temp = 0;
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for (std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it) {
    if (n % 96 == 0) {
      if (n > 0) {
        _mapDetIdNoise[temp] = vvnoise;
        _mapDetIdEff[temp] = veff;
        _bxmap[RPCDetId(it->dpid)] = it->time;

        veff.clear();
        vvnoise.clear();
        vvnoise.push_back((it->noise));
        veff.push_back((it->eff));
      } else if (n == 0) {
        vvnoise.push_back((it->noise));
        veff.push_back((it->eff));
        _bxmap[RPCDetId(it->dpid)] = it->time;
      }
    } else if (n == vnoise.size() - 1) {
      temp = it->dpid;
      vvnoise.push_back((it->noise));
      veff.push_back((it->eff));
      _mapDetIdNoise[temp] = vvnoise;
      _mapDetIdEff[temp] = veff;
    } else {
      temp = it->dpid;
      vvnoise.push_back((it->noise));
      veff.push_back((it->eff));
    }
    n++;
  }
}

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise,
                              const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize) {
  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp(vector<NoiseItem>, vector<ClusterSizeItem>)" << std::endl;

  uint32_t detId = 0, current_detId, this_detId;
  RPCDetId rpcId, current_rpcId, this_rpcId;
  const RPCRoll* current_roll = nullptr;
  const RPCRoll* this_roll = nullptr;
  unsigned int current_nStrips;

  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp :: ClusterSizeItem :: begin" << std::endl;
#ifdef EDM_ML_DEBUG
  std::stringstream sslogclsitem;
#endif
  // ### ClusterSizeItem #######################################################
  std::vector<RPCClusterSize::ClusterSizeItem>::const_iterator itCls;
  int clsCounter(1);
  std::vector<double> clsVect;
  // ### loop for New Format (120 entries)
  for (itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls) {
    clsVect.push_back(((double)(itCls->clusterSize)));
#ifdef EDM_ML_DEBUG
    sslogclsitem << " Push back clustersize = " << itCls->clusterSize << std::endl;
    sslogclsitem << "Filling cls in _mapDetCls[detId,clsVect] :: detId = " << detId;
    sslogclsitem << " --> will it be accepted? clsCounter = " << clsCounter << " accepted?";
    sslogclsitem << " New Format ::" << ((!(clsCounter % 120)) && (clsCounter != 0));  // <<std::endl;
    sslogclsitem << " Old Format ::" << ((!(clsCounter % 100)) && (clsCounter != 0));  // <<std::endl;
    sslogclsitem << std::endl;
#endif
    // New Format :: loop until 120
    if ((!(clsCounter % 120)) && (clsCounter != 0)) {
      detId = itCls->dpid;
      _mapDetClsMap[detId] = clsVect;
#ifdef EDM_ML_DEBUG
      std::stringstream LogDebugClsVectString;
      LogDebugClsVectString << "[";
      for (std::vector<double>::iterator itClsVect = clsVect.begin(); itClsVect != clsVect.end(); ++itClsVect) {
        LogDebugClsVectString << *itClsVect << ",";
      }
      LogDebugClsVectString << "]";
      std::string LogDebugClsVectStr = LogDebugClsVectString.str();
      LogDebug("RPCSimSetup") << "Filling clsVect in _mapDetCls[detId,clsVect] :: detId = " << RPCDetId(detId) << " = "
                              << detId << " clsVec = " << LogDebugClsVectStr;

      sslogclsitem << " --> New Method ";
      sslogclsitem << " --> saved in map " << std::endl;
      sslogclsitem << "Filling cls in _mapDetClsMap[detId,clsVect] :: detId = " << detId;
      sslogclsitem << " --> will it be accepted? clsCounter = " << clsCounter << " accepted? "
                   << ((!(clsCounter % 120)) && (clsCounter != 0)) << std::endl;
#endif
      clsVect.clear();
      clsCounter = 0;
    } else {
#ifdef EDM_ML_DEBUG
      sslogclsitem << " --> not saved in map " << std::endl;
#endif
    }
    ++clsCounter;
  }
  // ### loop for Old Format (100 entries)
  for (itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls) {
    clsVect.push_back(((double)(itCls->clusterSize)));
#ifdef EDM_ML_DEBUG
    sslogclsitem << " Push back clustersize = " << itCls->clusterSize << std::endl;
    sslogclsitem << "Filling cls in _mapDetClsMapLegacy[detId,clsVect] :: detId = " << detId;
    sslogclsitem << " --> will it be accepted? clsCounter = " << clsCounter << " accepted?";
    sslogclsitem << " New Format ::" << ((!(clsCounter % 120)) && (clsCounter != 0));  // <<std::endl;
    sslogclsitem << " Old Format ::" << ((!(clsCounter % 100)) && (clsCounter != 0));  // <<std::endl;
    sslogclsitem << std::endl;
#endif
    // Old Format :: same until 100
    if ((!(clsCounter % 100)) && (clsCounter != 0)) {
      detId = itCls->dpid;
      _mapDetClsMapLegacy[detId] = clsVect;
#ifdef EDM_ML_DEBUG
      std::stringstream LogDebugClsVectString;
      LogDebugClsVectString << "[";
      for (std::vector<double>::iterator itClsVect = clsVect.begin(); itClsVect != clsVect.end(); ++itClsVect) {
        LogDebugClsVectString << *itClsVect << ",";
      }
      LogDebugClsVectString << "]";
      std::string LogDebugClsVectStr = LogDebugClsVectString.str();
      LogDebug("RPCSimSetup") << "Filling clsVect in _mapDetClsLegacy[detId,clsVect] :: detId = " << RPCDetId(detId)
                              << " = " << detId << " clsVec = " << LogDebugClsVectStr;

      sslogclsitem << " --> Old Method ";
      sslogclsitem << " --> saved in map " << std::endl;
      sslogclsitem << "Filling cls in _mapDetClsMapLegacy[detId,clsVect] :: detId = " << detId;
      sslogclsitem << " --> will it be accepted? clsCounter = " << clsCounter << " accepted? "
                   << ((!(clsCounter % 120)) && (clsCounter != 0)) << std::endl;
#endif
      clsVect.clear();
      clsCounter = 0;
    } else {
#ifdef EDM_ML_DEBUG
      sslogclsitem << " --> not saved in map " << std::endl;
#endif
    }
    ++clsCounter;
  }
  // ###########################################################################
#ifdef EDM_ML_DEBUG
  std::string logclsitem = sslogclsitem.str();
  sslogclsitem.clear();
  LogDebug("RPCSimSetupClsLoopDetails") << logclsitem << std::endl;
  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp :: ClusterSizeItem :: end" << std::endl;

  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp :: NoiseItem :: begin" << std::endl;
  std::stringstream sslognoiseitem;
#endif
  // ### NoiseItem #############################################################
  unsigned int count_strips = 1;
#ifdef EDM_ML_DEBUG
  unsigned int count_all = 1;
#endif
  std::vector<float> vveff, vvnoise;

  // DetId to start with needs to be a DetId inside the Geometry used
  // Therefore loop on the NoiseItems and search for the first valid roll in the Geometry
  // Assign this as the DetId to start with (so called current_roll) and quit the loop
  bool quitLoop = false;
  current_detId = 0;
  current_nStrips = 0;  // current_rpcId = 0; current_roll = 0;
  for (std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end() && !quitLoop;
       ++it) {
    // roll associated to the conditions of this strip (iterator)
    current_detId = it->dpid;
    current_rpcId = RPCDetId(current_detId);
    // Test whether this roll (picked up from the conditions) is inside the RPC Geometry
    const RPCRoll* roll = theGeometry->roll(current_rpcId);
    if (roll == nullptr) {
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "Searching for first valid detid :: current_detId = " << current_detId;
      sslognoiseitem << " aka " << current_rpcId << " is not in current Geometry --> Skip " << std::endl;
#endif
      continue;
    } else {
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "Searching for first valid detid :: current_detId = " << current_detId;
      sslognoiseitem << " aka " << current_rpcId
                     << " is the first (valid) roll in the current Geometry --> Accept, Assign & Quit Loop"
                     << std::endl;
#endif
      current_roll = theGeometry->roll(current_rpcId);
      current_nStrips = current_roll->nstrips();
      quitLoop = true;
    }
  }

#ifdef EDM_ML_DEBUG
  sslognoiseitem << "Start Position ::            current_detId = " << current_detId << " aka " << current_rpcId;
  sslognoiseitem << " is a valid roll with pointer " << current_roll << " and has "
                 << (current_roll ? current_roll->nstrips() : 0) << " strips" << std::endl;
  sslognoiseitem << " -------------------------------------------------------------------------------------------------"
                    "------------------------------------ "
                 << std::endl;
#endif
  for (std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it) {
    // roll associated to the conditions of this strip (iterator)
    this_detId = it->dpid;
    this_rpcId = RPCDetId(this_detId);
    // Test whether this roll (picked up from the conditions) is inside the RPC Geometry
    const RPCRoll* roll = theGeometry->roll(this_rpcId);
    if (roll == nullptr) {
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "Inside Loop :: [" << std::setw(6) << count_all << "][" << std::setw(3) << count_strips
                     << "] :: this_detId = " << this_detId << " aka " << this_rpcId
                     << " which is not in current Geometry --> Skip " << std::endl;
#endif
      continue;
    }

    // Case 1 :: FIRST ENTRY
    // ---------------------
    if (this_detId == current_detId && count_strips == 1) {
      // fill bx in map
      _bxmap[current_detId] = it->time;
      // clear vectors
      vveff.clear();
      vvnoise.clear();
      // fill the vectors
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 1" << std::endl;
      sslognoiseitem << this_detId << " = " << this_rpcId << " with " << roll->nstrips() << " strips" << std::endl;
      sslognoiseitem << "[NoiseItem :: n = " << count_all
                     << "] Filling time in _bxmap[detId] :: detId = " << RPCDetId(it->dpid) << " time = " << it->time
                     << std::endl;
      sslognoiseitem << "First Value :: [" << std::setw(6) << count_all << "][" << std::setw(3) << count_strips
                     << "] :: this_detId = " << this_detId << " aka " << this_rpcId;
      sslognoiseitem << " Strip " << std::setw(3) << count_strips << " Noise = " << it->noise << " Hz/cm2" << std::endl;
      // update one counter
      ++count_all;
#endif
      // update the other counter
      ++count_strips;
    }
    // Case 2 :: 2ND ENTRY --> LAST-1 ENTRY
    // ------------------------------------
    else if (this_detId == current_detId && count_strips > 1 && count_strips < current_nStrips) {
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 2" << std::endl;
      sslognoiseitem << "Inside Loop :: [" << std::setw(6) << count_all << "][" << std::setw(3) << count_strips
                     << "] :: this_detId = " << this_detId << " aka " << this_rpcId;
      sslognoiseitem << " Strip " << std::setw(3) << count_strips << " Noise = " << it->noise << " Hz/cm2" << std::endl;
      // update one counter
      ++count_all;
#endif
      // fill the vectors
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
      // update the other counter
      ++count_strips;
    }

    // Case 3 :: LAST ENTRY
    // --------------------
    else if (this_detId == current_detId && count_strips == current_nStrips) {
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 3" << std::endl;
      sslognoiseitem << "Last Value ::  [" << std::setw(6) << count_all << "][" << std::setw(3) << count_strips
                     << "] :: this_detId = " << this_detId << " aka " << this_rpcId;
      sslognoiseitem << " Strip " << std::setw(3) << count_strips << " Noise = " << it->noise << " Hz/cm2" << std::endl;
      // update one counter
      ++count_all;
#endif
      // fill last value in the vector
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
      // update the other counter
      ++count_strips;
      // fill vectors into map
      _mapDetIdNoise[current_detId] = vvnoise;
      _mapDetIdEff[current_detId] = vveff;

#ifdef EDM_ML_DEBUG
      sslognoiseitem << " fill vectors into map" << std::endl;
      std::stringstream LogDebugNoiVectString, LogDebugEffVectString;
      LogDebugNoiVectString << "[";
      for (std::vector<float>::iterator itNoiVect = vvnoise.begin(); itNoiVect != vvnoise.end(); ++itNoiVect) {
        LogDebugNoiVectString << (*itNoiVect) << ",";
      }
      LogDebugNoiVectString << "]";
      std::string LogDebugNoiVectStr = LogDebugNoiVectString.str();
      LogDebugEffVectString << "[";
      for (std::vector<float>::iterator itEffVect = vveff.begin(); itEffVect != vveff.end(); ++itEffVect) {
        LogDebugEffVectString << (*itEffVect) << ",";
      }
      LogDebugEffVectString << "]";
      std::string LogDebugEffVectStr = LogDebugEffVectString.str();
      LogDebug("RPCSimSetup") << "Filling vvnoise in _mapDetIdNoise[detId] :: detId = " << RPCDetId(it->dpid) << " = "
                              << (RPCDetId(it->dpid)).rawId() << " vvnoise = " << LogDebugNoiVectStr;
      LogDebug("RPCSimSetup") << "Filling veff    in _mapDetIdEff[detId]   :: detId = " << RPCDetId(it->dpid) << " = "
                              << (RPCDetId(it->dpid)).rawId() << " veff    = " << LogDebugEffVectStr;
#endif
      // look for next different detId and rename it to the current_detId
      // at this point we skip all the conditions for the strips that are not in this roll
      // and we will go to the conditions for the first strip of the next roll
      bool next_detId_found = false;
#ifdef EDM_ML_DEBUG
      sslognoiseitem << "look for next different detId" << std::endl;
#endif
      while (next_detId_found == 0 && it != vnoise.end() - 1) {
        ++it;
        this_detId = it->dpid;
        this_rpcId = RPCDetId(this_detId);
        this_roll = theGeometry->roll(this_rpcId);
        if (!this_roll)
          continue;
#ifdef EDM_ML_DEBUG
        sslognoiseitem << "Inside While:: [" << std::setw(6) << count_all << "][" << std::setw(3) << count_strips
                       << "] :: this_detId = " << this_detId << " aka " << this_rpcId << " Noise = " << it->noise
                       << " Hz/cm2" << std::endl;
//	++count_all;
#endif
        ++count_strips;
        if (this_detId != current_detId) {
#ifdef EDM_ML_DEBUG
          sslognoiseitem << "Different detId is found ::                  " << this_detId << " aka " << this_rpcId
                         << " Noise = " << it->noise << " Hz/cm2";
#endif
          // next roll is found. update current_detId to this newly found detId
          // and update also the number of strips
          current_detId = this_detId;
          current_rpcId = RPCDetId(current_detId);
          next_detId_found = true;
          current_nStrips = (theGeometry->roll(current_rpcId))->nstrips();
#ifdef EDM_ML_DEBUG
          sslognoiseitem << " with " << current_nStrips << " strips" << std::endl;
#endif
          --it;  // subtract one, because at the end of the loop the iterator will be increased with one
                 // in fact the treatment for roll N stops when we find the first occurence of roll N+1
                 // however we want to start the treatment for roll N+1 with the first occurence of roll N+1
          // so the first entry of each new roll N+1 is manipulated twice in the loop (once as a stop, once as a start)
          // therefore we have to manipulate the iterator here, subtracting one, to treat again this entry
        }
      }
      // reset count_strips
      count_strips = 1;
    }
    // There should be no Case 4
    // -------------------------
    else {
    }
  }
  // ###########################################################################
#ifdef EDM_ML_DEBUG
  std::string lognoiseitem = sslognoiseitem.str();
  sslognoiseitem.clear();
  LogDebug("RPCSimSetupNoiseLoopDetails") << lognoiseitem << std::endl;
  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp :: NoiseItem :: end" << std::endl;

  LogDebug("RPCSimSetup") << "RPCSimSetUp::setRPCSetUp :: end" << std::endl;
#endif
}

const std::vector<float>& RPCSimSetUp::getNoise(uint32_t id) {
  std::map<uint32_t, std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  if (iter == _mapDetIdNoise.end()) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - no noise information for DetId\t" << id
                                        << std::endl;
  }
  LogDebug("RPCSimSetupChecks") << "All OK from RPCSimSetUp - noise information for DetId\t" << id << std::endl;
  return iter->second;
}

const std::vector<float>& RPCSimSetUp::getEff(uint32_t id) {
  std::map<uint32_t, std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
  if (iter == _mapDetIdEff.end()) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - no efficiency information for DetId\t" << id
                                        << std::endl;
  }

  RPCDetId rpcId = RPCDetId(id);
  const RPCRoll* roll = theGeometry->roll(rpcId);
  unsigned int numbStrips = roll->nstrips();

  if ((iter->second).size() < numbStrips) {
    LogDebug("RPCSimSetup") << "Exception from RPCSimSetUp - efficiency information in a wrong format for DetId\t" << id
                            << " aka " << RPCDetId(id) << std::endl;
    LogDebug("RPCSimSetup") << " number of strips in Conditions\t" << (iter->second).size()
                            << " number of strips in Geometry\t" << numbStrips << std::endl;
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - efficiency information in a wrong format for DetId\t" << id << std::endl;
  }

  return iter->second;
}

float RPCSimSetUp::getTime(uint32_t id) {
  RPCDetId rpcid(id);
  std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
  if (iter == _bxmap.end()) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - no timing information for rpcid.rawId()\t"
                                        << rpcid.rawId() << std::endl;
  }
  return iter->second;
}

const std::map<int, std::vector<double> >& RPCSimSetUp::getClsMap() {
  if (_clsMap.size() != 5) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - cluster size - a wrong format " << std::endl;
  }
  return _clsMap;
}

//const std::map<int, std::vector<double> >& RPCSimSetUp::getClsMap(uint32_t id)
const std::vector<double>& RPCSimSetUp::getCls(uint32_t id)  //legacy member function
{
  LogDebug("RPCSimSetupChecks") << "RPCSimSetUp::getCls" << std::endl;

  std::map<uint32_t, std::vector<double> >::iterator iter = _mapDetClsMapLegacy.find(id);
  if (iter == _mapDetClsMapLegacy.end()) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - no cluster size information for DetId\t" << id
                                        << std::endl;
  }
  if ((iter->second).size() != 100) {
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - _mapDetClsMapLegacy - cluster size information in a wrong format for DetId\t"
        << id << std::endl;
  }
  LogDebug("RPCSimSetupChecks")
      << "All OK from RPCSimSetUp - _mapDetClsMapLegacy - cluster size information for DetId\t" << id << std::endl;
  return iter->second;
}

const std::vector<double>& RPCSimSetUp::getAsymmetricClsDistribution(uint32_t id, uint32_t slice) {
  LogDebug("RPCSimSetupChecks") << "RPCSimSetUp::getAsymmetricClsDistribution" << std::endl;

  std::map<uint32_t, std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if (iter == _mapDetClsMap.end()) {
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - _mapDetClsMap - no cluster size information for DetId\t" << id << std::endl;
  }
  if ((iter->second).size() != 120) {
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - _mapDetClsMap - cluster size information in a wrong format for DetId\t" << id
        << std::endl;
  }
  //  return iter->second;

  std::vector<double> dataForAsymmCls = iter->second;
  if (slice > 4) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - slice variable not in the range" << std::endl;
  }

  _DetClsAsymmetric.clear();

  std::vector<double> clsFewStripsDistribution;
  std::vector<double> clsDistribution;
  std::vector<double> clsAccumulativeDistribution;

  std::map<int, std::vector<double> > mapSliceVsDistribution;

  const int slices = 5;
  const int distributionFewStrips = 24;

  double sliceVsFewStripsDistribution[slices][distributionFewStrips];

  for (int j = 0; j < distributionFewStrips; j++) {
    for (int i = 0; i < slices; i++) {
      sliceVsFewStripsDistribution[i][j] = dataForAsymmCls[j * slices + i];
    }
  }

  int i = slice;
  double sum = 0;
  int counter = 0;
  for (int j = 0; j < distributionFewStrips; j++) {
    counter++;
    sum += sliceVsFewStripsDistribution[i][j];
    if (counter % 4 == 0) {
      _DetClsAsymmetric.push_back(sum);
    }
  }
  return _DetClsAsymmetric;
}

const std::vector<double>& RPCSimSetUp::getAsymmetryForCls(uint32_t id, uint32_t slice, uint32_t cls) {
  LogDebug("RPCSimSetupChecks") << "RPCSimSetUp::getAsymmetryForCls" << std::endl;

  std::map<uint32_t, std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if (iter == _mapDetClsMap.end()) {
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - _mapDetClsMap - no cluster size information for DetId\t" << id << std::endl;
  }
  if ((iter->second).size() != 120) {
    throw cms::Exception("DataCorrupt")
        << "Exception from RPCSimSetUp - _mapDetClsMap - cluster size information in a wrong format for DetId\t" << id
        << '\t' << (iter->second).size() << std::endl;
  }

  std::vector<double> dataForAsymmCls = iter->second;

  if (slice > 4) {
    throw cms::Exception("DataCorrupt") << "Exception from RPCSimSetUp - slice variable not in the range" << std::endl;
  }

  _DetAsymmetryForCls.clear();

  std::vector<double> clsFewStripsDistribution;
  std::vector<double> clsDistribution;
  std::vector<double> clsAccumulativeDistribution;
  std::vector<double> clsDetAsymmetryForCls;
  clsDetAsymmetryForCls.clear();

  std::map<int, std::vector<double> > mapSliceVsDistribution;

  const int slices = 5;
  const int distributionFewStrips = 24;

  double sliceVsFewStripsDistribution[slices][distributionFewStrips];

  for (int j = 0; j < distributionFewStrips; j++) {
    for (int i = 0; i < slices; i++) {
      sliceVsFewStripsDistribution[i][j] = dataForAsymmCls[j * slices + i];
    }
  }

  int vector_lenght;
  switch (cls) {
    case 1:
    case 3:
    case 5:
      vector_lenght = 3;
      break;
    case 2:
    case 4:
      vector_lenght = 4;
      break;
    case 6:
    default:
      vector_lenght = 1;
      break;
  }

  float sum = 0;
  float value;
  for (int i = 0; i < vector_lenght; i++) {
    value = sliceVsFewStripsDistribution[slice][(cls - 1) * 4 + i];
    clsDetAsymmetryForCls.push_back(value);
    sum += value;
    //     LogDebug ("RPCSimSetup")<<"value\t"<<value<<std::endl;
    //    LogDebug ("RPCSimSetup")<<"sum\t"<<sum<<std::endl;
  }

  float accum = 0;
  for (int i = clsDetAsymmetryForCls.size() - 1; i > -1; i--) {
    accum += clsDetAsymmetryForCls[i];
    _DetAsymmetryForCls.push_back(accum / sum);
  }
  return _DetAsymmetryForCls;
}

RPCSimSetUp::~RPCSimSetUp() {}
