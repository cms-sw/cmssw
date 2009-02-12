#ifndef DataMixingHcalDigiWorkerProd_h
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Selector.h"

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "SimGeneral/DataMixingModule/plugins/HcalNoiseStorage.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingHcalDigiWorkerProd
    {
    public:

      DataMixingHcalDigiWorkerProd();

     /** standard constructor*/
      explicit DataMixingHcalDigiWorkerProd(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingHcalDigiWorkerProd();

      void putHcal(edm::Event &e,const edm::EventSetup& ES) ;
      void addHcalSignals(const edm::Event &e,const edm::EventSetup& ES); 
      void addHcalPileups(const int bcr, edm::Event*,unsigned int EventId,const edm::EventSetup& ES);


    private:
      // data specifiers

      // Hcal
      //      edm::InputTag HBHEdigiCollectionSig_; // secondary name given to collection of digis
      // edm::InputTag HOdigiCollectionSig_  ; // secondary name given to collection of digis
      //edm::InputTag HFdigiCollectionSig_  ; // secondary name given to collection of digis
      //edm::InputTag ZDCdigiCollectionSig_ ; // secondary name given to collection of digis
      edm::InputTag HBHEdigiCollectionPile_; // secondary name given to collection of digis
      edm::InputTag HOdigiCollectionPile_  ; // secondary name given to collection of digis
      edm::InputTag HFdigiCollectionPile_  ; // secondary name given to collection of digis
      edm::InputTag ZDCdigiCollectionPile_ ; // secondary name given to collection of digis
      std::string HBHEDigiCollectionDM_; // secondary name to be given to collection of digis
      std::string HODigiCollectionDM_  ; // secondary name to be given to collection of digis
      std::string HFDigiCollectionDM_  ; // secondary name to be given to collection of digis
      std::string ZDCDigiCollectionDM_ ; // secondary name to be given to collection of digis

      HcalDigitizer* myHcalDigitizer_;
      HcalNoiseStorage myHBHENoise_;
      HcalNoiseStorage myHONoise_;
      HcalNoiseStorage myHFNoise_;
      HcalNoiseStorage myZDCNoise_;

      std::vector<CaloSamples> HBHEDigiStore_;
      std::vector<CaloSamples> HODigiStore_;
      std::vector<CaloSamples> HFDigiStore_;
      std::vector<CaloSamples> ZDCDigiStore_;


      //      typedef std::multimap<DetId, HBHEDataFrame> HBHEDigiMap;
      //      typedef std::multimap<DetId, HFDataFrame>   HFDigiMap;
      //      typedef std::multimap<DetId, HODataFrame>   HODigiMap;
      //      typedef std::multimap<DetId, ZDCDataFrame>  ZDCDigiMap;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
