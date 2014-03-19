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
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ModuleCallingContext;

  class DataMixingHcalWorker
    {
    public:

      DataMixingHcalWorker();

     /** standard constructor*/
      explicit DataMixingHcalWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingHcalWorker();

      void putHcal(edm::Event &e) ;
      void addHcalSignals(const edm::Event &e); 
      void addHcalPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                          ModuleCallingContext const*);


    private:
      // data specifiers

      // Hcal
      edm::InputTag HBHErechitCollectionSig_; // secondary name given to collection of EB rechits
      edm::InputTag HOrechitCollectionSig_  ; // secondary name given to collection of EB rechits
      edm::InputTag HFrechitCollectionSig_  ; // secondary name given to collection of EB rechits
      edm::InputTag ZDCrechitCollectionSig_ ; // secondary name given to collection of EB rechits

      edm::InputTag HBHEPileRecHitInputTag_ ; // InputTag for HB RecHits for Pileup
      edm::InputTag HOPileRecHitInputTag_   ; // InputTag for HO RecHits for Pileup 
      edm::InputTag HFPileRecHitInputTag_   ; // InputTag for HF RecHits for Pileup 
      edm::InputTag ZDCPileRecHitInputTag_  ; // InputTag for ZDC RecHits for Pileup 

      edm::EDGetTokenT<HBHERecHitCollection> HBHERecHitToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<HORecHitCollection> HORecHitToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<HFRecHitCollection> HFRecHitToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<ZDCRecHitCollection> ZDCRecHitToken_ ;  // Token to retrieve information 

      edm::EDGetTokenT<HBHERecHitCollection> HBHERecHitPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<HORecHitCollection> HORecHitPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<HFRecHitCollection> HFRecHitPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<ZDCRecHitCollection> ZDCRecHitPToken_ ;  // Token to retrieve information 

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

      std::string label_;

    };
}//edm

#endif
