#ifndef DataMixingModule_h
#define SimDataMixingModule_h

/** \class DataMixingModule
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/
#include "Mixing/Base/interface/BMixingModule.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Selector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimGeneral/DataMixingModule/plugins/DataMixingEMWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingHcalWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingEMDigiWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingHcalDigiWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingMuonWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingSiStripWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingSiPixelWorker.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingModule : public BMixingModule
    {
    public:

      /** standard constructor*/
      explicit DataMixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingModule();

      virtual void beginJob(edm::EventSetup const&iSetup);

      virtual void setBcrOffset();
      virtual void setSourceOffset(const unsigned int is);      

    private:
      // data specifiers

      // Ecal
      // Rechits:
      edm::InputTag EBrechitCollectionSig_; // secondary name given to collection of EB rechits from "signal"
      edm::InputTag EErechitCollectionSig_; // secondary name given to collection of EE rechits
      edm::InputTag ESrechitCollectionSig_; // secondary name given to collection of EE rechits
      edm::InputTag EBrechitCollectionPile_; // secondary name given to collection of EB rechits from "pileup"
      edm::InputTag EErechitCollectionPile_; // secondary name given to collection of EE rechits
      edm::InputTag ESrechitCollectionPile_; // secondary name given to collection of EE rechits
      //output:
      std::string EBRecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EERecHitCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESRecHitCollectionDM_; // secondary name to be given to EE collection of hits
      //Digis:
      edm::InputTag EBdigiCollectionSig_; // secondary name given to collection of EB digis from "signal"
      edm::InputTag EEdigiCollectionSig_; // secondary name given to collection of EE digis
      edm::InputTag ESdigiCollectionSig_; // secondary name given to collection of EE digis
      edm::InputTag EBdigiCollectionPile_; // secondary name given to collection of EB digis from "pileup"
      edm::InputTag EEdigiCollectionPile_; // secondary name given to collection of EE digis
      edm::InputTag ESdigiCollectionPile_; // secondary name given to collection of EE digis
      //output
      std::string EBDigiCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EEDigiCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESDigiCollectionDM_; // secondary name to be given to EE collection of hits

      // Hcal
      // Rechits:
      edm::InputTag HBHErechitCollectionSig_; // secondary name given to collection of HBHE rechits from "signal"
      edm::InputTag HOrechitCollectionSig_  ; // secondary name given to collection of HO rechits
      edm::InputTag HFrechitCollectionSig_  ; // secondary name given to collection of HF rechits
      edm::InputTag ZDCrechitCollectionSig_ ; // secondary name given to collection of ZDC rechits
      edm::InputTag HBHErechitCollectionPile_; // secondary name given to collection of HBHE rechits from "pileup"
      edm::InputTag HOrechitCollectionPile_  ; // secondary name given to collection of HO rechits
      edm::InputTag HFrechitCollectionPile_  ; // secondary name given to collection of HF rechits
      edm::InputTag ZDCrechitCollectionPile_ ; // secondary name given to collection of ZDC rechits
      // output:
      std::string HBHERecHitCollectionDM_; // secondary name to be given to HBHE collection of hits
      std::string HORecHitCollectionDM_  ; // secondary name to be given to HO collection of hits
      std::string HFRecHitCollectionDM_  ; // secondary name to be given to HF collection of hits
      std::string ZDCRecHitCollectionDM_ ; // secondary name to be given to ZDC collection of hits
      // Digis:
      edm::InputTag HBHEdigiCollectionSig_; // secondary name given to collection of HBHE digis from "signal"
      edm::InputTag HOdigiCollectionSig_  ; // secondary name given to collection of HO digis
      edm::InputTag HFdigiCollectionSig_  ; // secondary name given to collection of HF digis
      edm::InputTag ZDCdigiCollectionSig_ ; // secondary name given to collection of ZDC digis
      edm::InputTag HBHEdigiCollectionPile_; // secondary name given to collection of HBHE digis from "pileup"
      edm::InputTag HOdigiCollectionPile_  ; // secondary name given to collection of HO digis
      edm::InputTag HFdigiCollectionPile_  ; // secondary name given to collection of HF digis
      edm::InputTag ZDCdigiCollectionPile_ ; // secondary name given to collection of ZDC digis
      // output:
      std::string HBHEDigiCollectionDM_; // secondary name to be given to HBHE collection of hits
      std::string HODigiCollectionDM_  ; // secondary name to be given to HO collection of hits
      std::string HFDigiCollectionDM_  ; // secondary name to be given to HF collection of hits
      std::string ZDCDigiCollectionDM_ ; // secondary name to be given to ZDC collection of hits

      // Muons
      edm::InputTag DTdigi_collectionSig_;      // secondary name given to collection of DT digis from "signal"
      edm::InputTag RPCdigi_collectionSig_;     // secondary name given to collection of RPC digis
      edm::InputTag CSCstripdigi_collectionSig_;// secondary name given to collection of CSC Strip digis
      edm::InputTag CSCwiredigi_collectionSig_; // secondary name given to collection of CSC wire digis
      edm::InputTag DTdigi_collectionPile_;      // secondary name given to collection of DT digis from "pileup"
      edm::InputTag RPCdigi_collectionPile_;     // secondary name given to collection of RPC digis
      edm::InputTag CSCstripdigi_collectionPile_;// secondary name given to collection of CSC Strip digis
      edm::InputTag CSCwiredigi_collectionPile_; // secondary name given to collection of CSC wire digis

      std::string DTDigiCollectionDM_;       // secondary name to be given to new DT digis
      std::string RPCDigiCollectionDM_;      // secondary name to be given to new RPC digis
      std::string CSCStripDigiCollectionDM_; // secondary name given to new collection of CSC Strip digis
      std::string CSCWireDigiCollectionDM_;  // secondary name given to new collection of CSC wire digis

      // SiStrips
      edm::InputTag Sistripdigi_collectionSig_ ; // secondary name given to collection of SiStrip digis from signal
      edm::InputTag Sistripdigi_collectionPile_ ; // secondary name given to collection of SiStrip digis from pileup
      std::string SiStripDigiCollectionDM_  ; // secondary name to be given to new SiStrip digis

      // SiPixels
      edm::InputTag pixeldigi_collectionSig_ ; // secondary name given to collection of SiPixel digis from signal
      edm::InputTag pixeldigi_collectionPile_ ; // secondary name given to collection of SiPixel digis from pileup
      std::string PixelDigiCollectionDM_  ; // secondary name to be given to new SiPixel digis

      // Submodules to handle the individual detectors

      DataMixingEMWorker *EMWorker_ ;
      DataMixingEMDigiWorker *EMDigiWorker_ ;
      bool MergeEMDigis_;

      // Hcal 
      
      DataMixingHcalWorker *HcalWorker_ ;
      DataMixingHcalDigiWorker *HcalDigiWorker_ ;
      bool MergeHcalDigis_;

      // Muons

      DataMixingMuonWorker *MuonWorker_ ;

      // Si-Strips

      DataMixingSiStripWorker *SiStripWorker_ ;

      // Pixels

      DataMixingSiPixelWorker *SiPixelWorker_ ;

      virtual void createnewEDProduct();
      virtual void getSubdetectorNames();

      // copies, with EventSetup
      virtual void put(edm::Event &e,const edm::EventSetup& ES) ;
      virtual void addSignals(const edm::Event &e, const edm::EventSetup& ES); 
      virtual void doPileUp(edm::Event &e,const edm::EventSetup& ES);
      virtual void addPileups(const int bcr, edm::Event*,unsigned int EventId,unsigned int worker,const edm::EventSetup& ES);
  

      // internally used information : subdetectors present in input
      std::vector<std::string> Subdetectors_;

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
