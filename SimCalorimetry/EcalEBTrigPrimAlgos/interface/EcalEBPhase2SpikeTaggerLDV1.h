#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTaggerLDV1_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTaggerLDV1_h

#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGSpikeTaggerParamsRcd.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTagger.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TPGSpikeTaggerParamsHelper.h"

/** \class EcalEBPhase2SpikeTaggerLDV1
   Linear discriminant (LD) ECAL spike tagger version 1
   Algorithm described in Section 9.3.1 of the CMS-TDR-015 (https://cds.cern.ch/record/2283187/)
*/

class EcalEBPhase2SpikeTaggerLDV1 : public EcalEBPhase2SpikeTagger {
public:
  EcalEBPhase2SpikeTaggerLDV1(edm::ConsumesCollector& cc, bool debug);

  bool process(const std::vector<int>& linInput) override;
  void getRecords(edm::EventSetup const& setup) override;
  void setParameters(EBDetId id, const EcalTPGCrystalStatus* ecaltpBadX) override;

private:
  edm::ESGetToken<EcalEBPhase2TPGSpikeTaggerParams, EcalEBPhase2TPGSpikeTaggerParamsRcd> spikeTaggerParamsToken_;
  unsigned int peakIdx_;
  float spikeThreshold_;
  std::vector<double> weights_;

  std::shared_ptr<EcalEBPhase2TPGSpikeTaggerParamsHelper> spikeTaggerParamsHelper_;

  float calcLD(std::vector<int> const& linInput) const;
  float calcRMinus1Poly(std::vector<int> const& linInput, float sMax) const;
};

#endif
