#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TPGSpikeTaggerParamsHelper_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2TPGSpikeTaggerParamsHelper_h

#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGSpikeTaggerParams.h"

class EcalEBPhase2TPGSpikeTaggerParamsHelper : public EcalEBPhase2TPGSpikeTaggerParams {
public:
  EcalEBPhase2TPGSpikeTaggerParamsHelper();
  EcalEBPhase2TPGSpikeTaggerParamsHelper(const EcalEBPhase2TPGSpikeTaggerParams &params);
  EcalEBPhase2TPGSpikeTaggerParamsHelper(const edm::ParameterSet &config);

  void createFromPSet(const edm::ParameterSet &config);

  // Global parameters
  unsigned int fwVersion() const;
  void setFwVersion(const unsigned int fwVersion);

  unsigned int peakSampleIndex(const EBDetId &detId) const;
  void setPeakSampleIndex(const EBDetId &detId, const unsigned int soi);

  // Spike tagger LD parameters
  double spikeTaggerLdThreshold(const EBDetId &detId) const;
  void setSpikeTaggerLdThreshold(const EBDetId &detId, const double &thr);

  std::vector<double> spikeTaggerLdWeights(const EBDetId &detId) const;
  void setSpikeTaggerLdWeights(const EBDetId &detId, const std::vector<double> &weights);

  // print parameters to stream:
  void print(std::ostream &out) const;
  friend std::ostream &operator<<(std::ostream &out, const EcalEBPhase2TPGSpikeTaggerParamsHelper &params);

private:
  // Defines the content of each node
  // New nodes can only be added before NUM_NODES
  enum EcalSpikeTaggerParamNode { kGlobalAlgoParams = 0, NUM_NODES };

  // Defines the content of each crystal node
  // New nodes can only be added before NUM_CRYSTAL_NODES
  enum EcalSpikeTaggerParamCrystalNode {
    kCrystalSpikeTaggerLdParams = 0,
    kCrystalSpikeTaggerLdWeights,
    NUM_CRYSTAL_NODES
  };

  // index of variabe inside a node vector
  enum DIdx { kSpikeThreshold = 0 };
  enum UIdx { kFwVersion = 0, kPeakSampleIndex = 0 };
  enum IIdx {};
  enum SIdx {};

  void setPerCrystalSpikeTaggerParams(const std::vector<edm::ParameterSet> &pSets);
  void parseCrystalRange(const std::string &rangeStr, int &iMin, int &iMax, const bool isEta = true);
};
#endif
