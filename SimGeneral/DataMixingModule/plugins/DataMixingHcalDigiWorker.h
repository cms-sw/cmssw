#ifndef DataMixingHcalDigiWorker_h
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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"


#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingHcalDigiWorker
    {
    public:

      DataMixingHcalDigiWorker();

     /** standard constructor*/
      explicit DataMixingHcalDigiWorker(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingHcalDigiWorker();

      void putHcal(edm::Event &e,const edm::EventSetup& ES) ;
      void addHcalSignals(const edm::Event &e,const edm::EventSetup& ES); 
      void addHcalPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,const edm::EventSetup& ES);


    private:
      // data specifiers

      // Hcal
      edm::InputTag HBHEdigiCollectionSig_; // secondary name given to collection of digis
      edm::InputTag HOdigiCollectionSig_  ; // secondary name given to collection of digis
      edm::InputTag HFdigiCollectionSig_  ; // secondary name given to collection of digis
      edm::InputTag ZDCdigiCollectionSig_ ; // secondary name given to collection of digis

      edm::InputTag HBHEPileInputTag_; // InputTag for Pileup Digis collection  
      edm::InputTag HOPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag HFPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag ZDCPileInputTag_ ; // InputTag for Pileup Digis collection

      std::string HBHEDigiCollectionDM_; // secondary name to be given to collection of digis
      std::string HODigiCollectionDM_  ; // secondary name to be given to collection of digis
      std::string HFDigiCollectionDM_  ; // secondary name to be given to collection of digis
      std::string ZDCDigiCollectionDM_ ; // secondary name to be given to collection of digis

      typedef std::multimap<DetId, CaloSamples> HBHEDigiMap;
      typedef std::multimap<DetId, CaloSamples> HFDigiMap;
      typedef std::multimap<DetId, CaloSamples> HODigiMap;
      typedef std::multimap<DetId, CaloSamples> ZDCDigiMap;

      //      typedef std::multimap<DetId, HBHEDataFrame> HBHEDigiMap;
      //      typedef std::multimap<DetId, HFDataFrame>   HFDigiMap;
      //      typedef std::multimap<DetId, HODataFrame>   HODigiMap;
      //      typedef std::multimap<DetId, ZDCDataFrame>  ZDCDigiMap;

      HBHEDigiMap HBHEDigiStorage_;
      HFDigiMap   HFDigiStorage_;
      HODigiMap   HODigiStorage_;
      ZDCDigiMap  ZDCDigiStorage_;

      bool DoZDC_;

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

    };
}//edm

#endif
