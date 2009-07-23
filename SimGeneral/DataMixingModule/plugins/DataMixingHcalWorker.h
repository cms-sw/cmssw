#ifndef DataMixingHcalWorker_h
#define SimDataMixingHcalWorker_h

/** \class DataMixingHcalWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the Hcal information
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
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingHcalWorker
    {
    public:

      DataMixingHcalWorker();

     /** standard constructor*/
      explicit DataMixingHcalWorker(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingHcalWorker();

      void putHcal(edm::Event &e) ;
      void addHcalSignals(const edm::Event &e); 
      void addHcalPileups(const int bcr, edm::Event*,unsigned int EventId);


    private:
      // data specifiers

      // Hcal
      edm::InputTag HBHErechitCollectionSig_; // secondary name given to collection of EB rechits
      edm::InputTag HOrechitCollectionSig_  ; // secondary name given to collection of EB rechits
      edm::InputTag HFrechitCollectionSig_  ; // secondary name given to collection of EB rechits
      edm::InputTag ZDCrechitCollectionSig_ ; // secondary name given to collection of EB rechits
      edm::InputTag HBHErechitCollectionPile_; // secondary name given to collection of EB rechits
      edm::InputTag HOrechitCollectionPile_  ; // secondary name given to collection of EB rechits
      edm::InputTag HFrechitCollectionPile_  ; // secondary name given to collection of EB rechits
      edm::InputTag ZDCrechitCollectionPile_ ; // secondary name given to collection of EB rechits
      std::string HBHERecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string HORecHitCollectionDM_  ; // secondary name to be given to EB collection of hits
      std::string HFRecHitCollectionDM_  ; // secondary name to be given to EB collection of hits
      std::string ZDCRecHitCollectionDM_ ; // secondary name to be given to EB collection of hits

      typedef std::multimap<DetId, HBHERecHit> HBHERecHitMap;
      typedef std::multimap<DetId, HFRecHit>   HFRecHitMap;
      typedef std::multimap<DetId, HORecHit>   HORecHitMap;
      typedef std::multimap<DetId, ZDCRecHit>  ZDCRecHitMap;

      HBHERecHitMap HBHERecHitStorage_;
      HFRecHitMap   HFRecHitStorage_;
      HORecHitMap   HORecHitStorage_;
      ZDCRecHitMap  ZDCRecHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
