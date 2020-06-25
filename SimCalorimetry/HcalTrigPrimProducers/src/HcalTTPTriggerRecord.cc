#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTTPTriggerRecord.h"

#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalTTPTriggerRecord::HcalTTPTriggerRecord(const edm::ParameterSet& ps) {
  tok_ttp_ = consumes<HcalTTPDigiCollection>(ps.getParameter<edm::InputTag>("ttpDigiCollection"));
  ttpBits_ = ps.getParameter<std::vector<unsigned int> >("ttpBits");
  names_ = ps.getParameter<std::vector<std::string> >("ttpBitNames");

  produces<L1GtTechnicalTriggerRecord>();
}

HcalTTPTriggerRecord::~HcalTTPTriggerRecord() {}

void HcalTTPTriggerRecord::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  std::vector<L1GtTechnicalTrigger> vecTT(ttpBits_.size());

  // Get Inputs
  edm::Handle<HcalTTPDigiCollection> ttpDigiCollection;
  e.getByToken(tok_ttp_, ttpDigiCollection);

  if (!ttpDigiCollection.failedToGet()) {
    const HcalTTPDigiCollection* ttpDigis = ttpDigiCollection.product();
    uint8_t ttpResults = ttpDigis->begin()->triggerOutput();
    bool bit8 = ttpResults & 0x1;
    bool bit9 = ttpResults & 0x2;
    bool bit10 = ttpResults & 0x4;

    for (unsigned int i = 0; i < ttpBits_.size(); i++) {
      bool bitValue = false;
      if (ttpBits_.at(i) == 8)
        bitValue = bit8;
      if (ttpBits_.at(i) == 9)
        bitValue = bit9;
      if (ttpBits_.at(i) == 10)
        bitValue = bit10;
      vecTT.at(i) = L1GtTechnicalTrigger(names_.at(i), ttpBits_.at(i), 0, bitValue);
    }
  } else {
    vecTT.clear();
  }

  // Put output into event
  std::unique_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord());
  output->setGtTechnicalTrigger(vecTT);
  e.put(std::move(output));
}
