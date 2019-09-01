#ifndef SimDataMixingHcalDigiWorkerProd_h
#define SimDataMixingHcalDigiWorkerProd_h

/** \class DataMixingHcalDigiWorkerProd
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
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSignalGenerator.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigiProducer.h"
#include "SimGeneral/DataMixingModule/plugins/HcalNoiseStorage.h"

#include <map>
#include <string>
#include <vector>

namespace edm {
  class ConsumesCollector;
  class ModuleCallingContext;

  class DataMixingHcalDigiWorkerProd {
  public:
    /** standard constructor*/
    explicit DataMixingHcalDigiWorkerProd(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);

    /**Default destructor*/
    virtual ~DataMixingHcalDigiWorkerProd();

    void putHcal(edm::Event &e, const edm::EventSetup &ES);
    void addHcalSignals(const edm::Event &e, const edm::EventSetup &ES);
    void addHcalPileups(const int bcr,
                        const edm::EventPrincipal *,
                        unsigned int EventId,
                        const edm::EventSetup &ES,
                        edm::ModuleCallingContext const *);

    void beginRun(const edm::Run &run, const edm::EventSetup &ES);
    void initializeEvent(const edm::Event &e, const edm::EventSetup &ES);

  private:
    // data specifiers

    // Hcal
    edm::InputTag HBHEPileInputTag_;     // InputTag for Pileup Digis collection
    edm::InputTag HOPileInputTag_;       // InputTag for Pileup Digis collection
    edm::InputTag HFPileInputTag_;       // InputTag for Pileup Digis collection
    edm::InputTag ZDCPileInputTag_;      // InputTag for Pileup Digis collection
    edm::InputTag QIE10PileInputTag_;    // InputTag for Pileup Digis collection
    edm::InputTag QIE11PileInputTag_;    // InputTag for Pileup Digis collection
    std::string HBHEDigiCollectionDM_;   // secondary name to be given to collection
                                         // of digis
    std::string HODigiCollectionDM_;     // secondary name to be given to collection of digis
    std::string HFDigiCollectionDM_;     // secondary name to be given to collection of digis
    std::string ZDCDigiCollectionDM_;    // secondary name to be given to collection of digis
    std::string QIE10DigiCollectionDM_;  // secondary name to be given to
                                         // collection of digis
    std::string QIE11DigiCollectionDM_;  // secondary name to be given to
                                         // collection of digis

    edm::EDGetTokenT<HBHEDigitizerTraits::DigiCollection> tok_hbhe_;
    edm::EDGetTokenT<HODigitizerTraits::DigiCollection> tok_ho_;
    edm::EDGetTokenT<HFDigitizerTraits::DigiCollection> tok_hf_;
    edm::EDGetTokenT<ZDCDigitizerTraits::DigiCollection> tok_zdc_;
    edm::EDGetTokenT<HcalQIE10DigitizerTraits::DigiCollection> tok_qie10_;
    edm::EDGetTokenT<HcalQIE11DigitizerTraits::DigiCollection> tok_qie11_;

    HcalDigiProducer *myHcalDigitizer_;
    HBHESignalGenerator theHBHESignalGenerator;
    HOSignalGenerator theHOSignalGenerator;
    HFSignalGenerator theHFSignalGenerator;
    ZDCSignalGenerator theZDCSignalGenerator;
    QIE10SignalGenerator theQIE10SignalGenerator;
    QIE11SignalGenerator theQIE11SignalGenerator;

    std::string label_;
  };
}  // namespace edm

#endif  // SimDataMixingHcalDigiWorkerProd_h
