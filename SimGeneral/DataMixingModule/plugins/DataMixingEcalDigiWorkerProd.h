#ifndef DataMixingEcalDigiWorkerProd_h
#define SimDataMixingEcalDigiWorkerProd_h

/** \class DataMixingEcalDigiWorkerProd
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the Ecal information at Digi level
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
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGenerator.h"
#include "SimGeneral/DataMixingModule/plugins/EcalNoiseStorage.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ConsumesCollector;
  class ModuleCallingContext;

  class DataMixingEcalDigiWorkerProd
    {
    public:

     /** standard constructor*/
      explicit DataMixingEcalDigiWorkerProd(const edm::ParameterSet& ps, edm::ConsumesCollector& iC);

      /**Default destructor*/
      virtual ~DataMixingEcalDigiWorkerProd();

      void putEcal(edm::Event &e,const edm::EventSetup& ES);
      void addEcalSignals(const edm::Event &e,const edm::EventSetup& ES); 
      void addEcalPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                          const edm::EventSetup& ES, edm::ModuleCallingContext const*);

    // set tokens for data access
    void setEBAccess( edm::EDGetTokenT<EBDigitizerTraits::DigiCollection> tok) { tok_eb_ = tok; }
    void setEEAccess( edm::EDGetTokenT<EEDigitizerTraits::DigiCollection> tok) { tok_ee_ = tok; }
    void setESAccess( edm::EDGetTokenT<ESDigitizerTraits::DigiCollection> tok) { tok_es_ = tok; }

    void beginRun(const edm::EventSetup& ES);
    void initializeEvent(const edm::Event &e, const edm::EventSetup& ES);

    private:
      // data specifiers

      // Ecal
      //      edm::InputTag EBdigiCollectionSig_; // secondary name given to collection of digis
      // edm::InputTag EEdigiCollectionSig_  ; // secondary name given to collection of digis
      //edm::InputTag ESdigiCollectionSig_  ; // secondary name given to collection of digis
      //edm::InputTag ZDCdigiCollectionSig_ ; // secondary name given to collection of digis
      edm::InputTag EBPileInputTag_; // InputTag for Pileup Digis collection  
      edm::InputTag EEPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag ESPileInputTag_  ; // InputTag for Pileup Digis collection

      std::string EBDigiCollectionDM_; // secondary name to be given to collection of digis
      std::string EEDigiCollectionDM_  ; // secondary name to be given to collection of digis
      std::string ESDigiCollectionDM_  ; // secondary name to be given to collection of digis

      edm::EDGetTokenT<EBDigitizerTraits::DigiCollection> tok_eb_;
      edm::EDGetTokenT<EEDigitizerTraits::DigiCollection> tok_ee_;
      edm::EDGetTokenT<ESDigitizerTraits::DigiCollection> tok_es_;

      const double m_EBs25notCont;
      const double m_EEs25notCont;
      const double m_peToABarrel;
      const double m_peToAEndcap;


      EcalDigiProducer* myEcalDigitizer_;
      EBSignalGenerator theEBSignalGenerator;
      EESignalGenerator theEESignalGenerator;
      ESSignalGenerator theESSignalGenerator;

      std::string label_;

    };
}//edm

#endif
