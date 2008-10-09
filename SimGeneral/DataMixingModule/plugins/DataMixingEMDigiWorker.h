#ifndef DataMixingEMDigiWorker_h
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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Selector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"


#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingEMDigiWorker
    {
    public:

      DataMixingEMDigiWorker();

     /** standard constructor*/
      explicit DataMixingEMDigiWorker(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingEMDigiWorker();

      void putEM(edm::Event &e,const edm::EventSetup& ES) ;
      void addEMSignals(const edm::Event &e,const edm::EventSetup& ES); 
      void addEMPileups(const int bcr, edm::Event*,unsigned int EventId,const edm::EventSetup& ES);


    private:
      // data specifiers

      edm::InputTag EBProducerSig_; // primary? name given to collection of EB calib digis
      edm::InputTag EEProducerSig_; // primary? name given to collection of EE calib digis
      edm::InputTag ESProducerSig_; // primary? name given to collection of ES calib digis
      edm::InputTag EBProducerPile_; // primary? name given to collection of EB calib digis
      edm::InputTag EEProducerPile_; // primary? name given to collection of EE calib digis
      edm::InputTag ESProducerPile_; // primary? name given to collection of ES calib digis

      edm::InputTag EBdigiCollectionSig_; // secondary name given to collection of EB calib digis
      edm::InputTag EEdigiCollectionSig_; // secondary name given to collection of EE calib digis
      edm::InputTag ESdigiCollectionSig_; // secondary name given to collection of ES calib digis
      edm::InputTag EBdigiCollectionPile_; // secondary name given to collection of EB calib digis
      edm::InputTag EEdigiCollectionPile_; // secondary name given to collection of EE calib digis
      edm::InputTag ESdigiCollectionPile_; // secondary name given to collection of ES calib digis
      std::string EBDigiCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EEDigiCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESDigiCollectionDM_; // secondary name to be given to ES collection of hits

      typedef std::multimap<DetId, EBDataFrame> EBDigiMap;
      typedef std::multimap<DetId, EEDataFrame> EEDigiMap;
      typedef std::multimap<DetId, ESDataFrame> ESDigiMap;

      EBDigiMap EBDigiStorage_;
      EEDigiMap EEDigiStorage_;
      ESDigiMap ESDigiStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
