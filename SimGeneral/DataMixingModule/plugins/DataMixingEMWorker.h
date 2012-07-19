#ifndef DataMixingEMWorker_h
#define SimDataMixingEMWorker_h

/** \class DataMixingEMWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the EM information
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingEMWorker
    {
    public:

      DataMixingEMWorker();

     /** standard constructor*/
      explicit DataMixingEMWorker(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingEMWorker();

      void putEM(edm::Event &e) ;
      void addEMSignals(const edm::Event &e); 
      void addEMPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId);


    private:
      // data specifiers

      edm::InputTag EBProducerSig_; // primary? name given to collection of EB calib rechits
      edm::InputTag EEProducerSig_; // primary? name given to collection of EE calib rechits
      edm::InputTag ESProducerSig_; // primary? name given to collection of ES calib rechits

      edm::InputTag EBrechitCollectionSig_; // secondary name given to collection of EB calib rechits
      edm::InputTag EErechitCollectionSig_; // secondary name given to collection of EE calib rechits
      edm::InputTag ESrechitCollectionSig_; // secondary name given to collection of ES calib rechits

      edm::InputTag EBPileRecHitInputTag_; // full InputTag for pileup EB calib rechits
      edm::InputTag EEPileRecHitInputTag_; // full InputTag for pileup EE calib rechits
      edm::InputTag ESPileRecHitInputTag_; // full InputTag for pileup ES calib rechits




      std::string EBRecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EERecHitCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESRecHitCollectionDM_; // secondary name to be given to ES collection of hits

      typedef std::multimap<DetId, EcalRecHit> EBRecHitMap;
      typedef std::multimap<DetId, EcalRecHit> EERecHitMap;
      typedef std::multimap<DetId, EcalRecHit> ESRecHitMap;

      EBRecHitMap EBRecHitStorage_;
      EERecHitMap EERecHitStorage_;
      ESRecHitMap ESRecHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

    };
}//edm

#endif
