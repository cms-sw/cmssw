#ifndef SimDataMixingEMDigiWorker_h
#define SimDataMixingEMDigiWorker_h

/** \class DataMixingEMDigiWorker
 *
 * DataMixingModule is the EDProducer subclass
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the EM information at the Digi level
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <map>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class DataMixingEMDigiWorker {
  public:
    DataMixingEMDigiWorker();

    /** standard constructor*/
    explicit DataMixingEMDigiWorker(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);

    /**Default destructor*/
    virtual ~DataMixingEMDigiWorker();

    void putEM(edm::Event &e, const edm::EventSetup &ES);
    void addEMSignals(const edm::Event &e, const edm::EventSetup &ES);
    void addEMPileups(const int bcr,
                      const edm::EventPrincipal *,
                      unsigned int EventId,
                      const edm::EventSetup &ES,
                      ModuleCallingContext const *);

  private:
    /// retrieve pedestals for that detid [0]=g12, [1]=g6, [2]=g12
    const std::vector<float> GetPedestals(const edm::EventSetup &ES, const DetId &detid);

    /// retrieve gain ratios for that detid [0]=g12, [1]=g6, [2]=g12
    const std::vector<float> GetGainRatios(const edm::EventSetup &ES, const DetId &detid);

    // data specifiers

    edm::InputTag EBProducerSig_;  // primary? name given to collection of EB calib digis
    edm::InputTag EEProducerSig_;  // primary? name given to collection of EE calib digis
    edm::InputTag ESProducerSig_;  // primary? name given to collection of ES calib digis

    edm::InputTag EBdigiCollectionSig_;  // secondary name given to collection of
                                         // EB calib digis
    edm::InputTag EEdigiCollectionSig_;  // secondary name given to collection of
                                         // EE calib digis
    edm::InputTag ESdigiCollectionSig_;  // secondary name given to collection of
                                         // ES calib digis

    edm::InputTag EBPileInputTag_;  // complete input tag for EB pileup digis
    edm::InputTag EEPileInputTag_;  // complete input tag for EE pileup digis
    edm::InputTag ESPileInputTag_;  // complete input tag for ES pileup digis

    edm::EDGetTokenT<EBDigiCollection> EBDigiToken_;  // Token to retrieve information
    edm::EDGetTokenT<EEDigiCollection> EEDigiToken_;  // Token to retrieve information
    edm::EDGetTokenT<ESDigiCollection> ESDigiToken_;  // Token to retrieve information

    edm::EDGetTokenT<EBDigiCollection> EBDigiPileToken_;  // Token to retrieve information
    edm::EDGetTokenT<EEDigiCollection> EEDigiPileToken_;  // Token to retrieve information
    edm::EDGetTokenT<ESDigiCollection> ESDigiPileToken_;  // Token to retrieve information

    std::string EBDigiCollectionDM_;  // secondary name to be given to EB
                                      // collection of hits
    std::string EEDigiCollectionDM_;  // secondary name to be given to EE
                                      // collection of hits
    std::string ESDigiCollectionDM_;  // secondary name to be given to ES
                                      // collection of hits

    typedef std::multimap<DetId, EBDataFrame> EBDigiMap;
    typedef std::multimap<DetId, EEDataFrame> EEDigiMap;
    typedef std::multimap<DetId, ESDataFrame> ESDigiMap;

    EBDigiMap EBDigiStorage_;
    EEDigiMap EEDigiStorage_;
    ESDigiMap ESDigiStorage_;

    //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

    std::string label_;
  };
}  // namespace edm

#endif  // SimDataMixingEMDigiWorker_h
