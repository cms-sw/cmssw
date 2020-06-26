#ifndef HcalTrigPrimProducers_HcalTTPDigiProducer_h
#define HcalTrigPrimProducers_HcalTTPDigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"

class HcalTTPDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit HcalTTPDigiProducer(const edm::ParameterSet& ps);
  ~HcalTTPDigiProducer() override = default;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  bool isMasked(HcalDetId id);
  bool decision(int nP, int nM, int bit);

  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::ESGetToken<HcalTPGCoder, HcalTPGRecord> tok_tpgCoder_;
  std::vector<unsigned int> maskedChannels_;
  std::string bit_[4];
  int calc_[4];
  int nHits_[4], nHFp_[4], nHFm_[4];
  char pReq_[4], mReq_[4], pmLogic_[4];
  int id_, samples_, presamples_;
  int fwAlgo_;
  int iEtaMin_, iEtaMax_;
  unsigned int threshold_;

  int SoI_;

  static const int inputs_[];
};

#endif
