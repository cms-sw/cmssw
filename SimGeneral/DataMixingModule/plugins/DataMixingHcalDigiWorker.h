#ifndef SimDataMixingHcalDigiWorker_h
#define SimDataMixingHcalDigiWorker_h

/** \class DataMixingHcalDigiWorker
 *
 * DataMixingModule is the EDProducer subclass
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the Hcal information at Digi level
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version June 2008
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <map>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class DataMixingHcalDigiWorker {
  public:
    DataMixingHcalDigiWorker();

    /** standard constructor*/
    explicit DataMixingHcalDigiWorker(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);

    /**Default destructor*/
    virtual ~DataMixingHcalDigiWorker();

    void putHcal(edm::Event &e, const edm::EventSetup &ES);
    void addHcalSignals(const edm::Event &e, const edm::EventSetup &ES);
    void addHcalPileups(const int bcr,
                        const edm::EventPrincipal *,
                        unsigned int EventId,
                        const edm::EventSetup &ES,
                        ModuleCallingContext const *);

  private:
    // data specifiers

    // Hcal
    edm::InputTag HBHEdigiCollectionSig_;   // secondary name given to collection of digis
    edm::InputTag HOdigiCollectionSig_;     // secondary name given to collection of digis
    edm::InputTag HFdigiCollectionSig_;     // secondary name given to collection of digis
    edm::InputTag ZDCdigiCollectionSig_;    // secondary name given to collection of digis
    edm::InputTag QIE10digiCollectionSig_;  // secondary name given to collection of digis
    edm::InputTag QIE11digiCollectionSig_;  // secondary name given to collection of digis

    edm::InputTag HBHEPileInputTag_;   // InputTag for Pileup Digis collection
    edm::InputTag HOPileInputTag_;     // InputTag for Pileup Digis collection
    edm::InputTag HFPileInputTag_;     // InputTag for Pileup Digis collection
    edm::InputTag ZDCPileInputTag_;    // InputTag for Pileup Digis collection
    edm::InputTag QIE10PileInputTag_;  // InputTag for Pileup Digis collection
    edm::InputTag QIE11PileInputTag_;  // InputTag for Pileup Digis collection

    edm::EDGetTokenT<HBHEDigiCollection> HBHEDigiToken_;    // Token to retrieve information
    edm::EDGetTokenT<HODigiCollection> HODigiToken_;        // Token to retrieve information
    edm::EDGetTokenT<HFDigiCollection> HFDigiToken_;        // Token to retrieve information
    edm::EDGetTokenT<ZDCDigiCollection> ZDCDigiToken_;      // Token to retrieve information
    edm::EDGetTokenT<QIE10DigiCollection> QIE10DigiToken_;  // Token to retrieve information
    edm::EDGetTokenT<QIE11DigiCollection> QIE11DigiToken_;  // Token to retrieve information

    edm::EDGetTokenT<HBHEDigiCollection> HBHEDigiPToken_;    // Token to retrieve information
    edm::EDGetTokenT<HODigiCollection> HODigiPToken_;        // Token to retrieve information
    edm::EDGetTokenT<HFDigiCollection> HFDigiPToken_;        // Token to retrieve information
    edm::EDGetTokenT<ZDCDigiCollection> ZDCDigiPToken_;      // Token to retrieve information
    edm::EDGetTokenT<QIE10DigiCollection> QIE10DigiPToken_;  // Token to retrieve information
    edm::EDGetTokenT<QIE11DigiCollection> QIE11DigiPToken_;  // Token to retrieve information

    std::string HBHEDigiCollectionDM_;   // secondary name to be given to collection
                                         // of digis
    std::string HODigiCollectionDM_;     // secondary name to be given to collection of digis
    std::string HFDigiCollectionDM_;     // secondary name to be given to collection of digis
    std::string ZDCDigiCollectionDM_;    // secondary name to be given to collection of digis
    std::string QIE10DigiCollectionDM_;  // secondary name to be given to
                                         // collection of digis
    std::string QIE11DigiCollectionDM_;  // secondary name to be given to
                                         // collection of digis

    typedef std::multimap<DetId, CaloSamples> HBHEDigiMap;
    typedef std::multimap<DetId, CaloSamples> HFDigiMap;
    typedef std::multimap<DetId, CaloSamples> HODigiMap;
    typedef std::multimap<DetId, CaloSamples> ZDCDigiMap;
    typedef std::multimap<DetId, CaloSamples> QIE10DigiMap;
    typedef std::multimap<DetId, CaloSamples> QIE11DigiMap;

    HBHEDigiMap HBHEDigiStorage_;
    HFDigiMap HFDigiStorage_;
    HODigiMap HODigiStorage_;
    ZDCDigiMap ZDCDigiStorage_;
    QIE10DigiMap QIE10DigiStorage_;
    QIE11DigiMap QIE11DigiStorage_;

    bool DoZDC_;

    std::string label_;
  };
}  // namespace edm

#endif  // SimDataMixingHcalDigiWorker_h
