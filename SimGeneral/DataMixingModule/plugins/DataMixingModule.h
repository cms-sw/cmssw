#ifndef SimGeneral_DataMixingModule_DataMixingModule_h
#define SimGeneral_DataMixingModule_DataMixingModule_h

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
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "SimGeneral/DataMixingModule/plugins/DataMixingEMWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingHcalWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingEMDigiWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingHcalDigiWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingHcalDigiWorkerProd.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingMuonWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingSiStripWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingSiStripRawWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingSiPixelWorker.h"
#include "SimGeneral/DataMixingModule/plugins/DataMixingPileupCopy.h"

#include <map>
#include <vector>
#include <string>

namespace edm {

  class ModuleCallingContext;

  class DataMixingModule : public BMixingModule
    {
    public:

      /** standard constructor*/
      explicit DataMixingModule(const edm::ParameterSet& ps, MixingCache::Config const* globalConf);

      /**Default destructor*/
      ~DataMixingModule() override;

      // copies, with EventSetup
      void checkSignal(const edm::Event &e) override {}; 
      void createnewEDProduct() override {}
      void addSignals(const edm::Event &e, const edm::EventSetup& ES) override; 
      void doPileUp(edm::Event &e,const edm::EventSetup& ES) override;
      void put(edm::Event &e,const edm::EventSetup& ES) override ;

      void initializeEvent(edm::Event const& e, edm::EventSetup const& eventSetup) override;
      void beginRun(edm::Run const& run, edm::EventSetup const& eventSetup) override;
      void pileWorker(const edm::EventPrincipal&, int bcr, int EventId,const edm::EventSetup& ES, ModuleCallingContext const*);
      //virtual void beginJob();
      //virtual void endJob();
      void beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;
      void endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;
      void endRun(const edm::Run& r, const edm::EventSetup& setup) override;



    private:
      // data specifiers

      // Ecal
      //output:
      std::string EBRecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EERecHitCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESRecHitCollectionDM_; // secondary name to be given to EE collection of hits
      //Digis:
      //output
      std::string EBDigiCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EEDigiCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESDigiCollectionDM_; // secondary name to be given to EE collection of hits

      // Hcal
      // Rechits:
      // output:
      std::string HBHERecHitCollectionDM_; // secondary name to be given to HBHE collection of hits
      std::string HORecHitCollectionDM_  ; // secondary name to be given to HO collection of hits
      std::string HFRecHitCollectionDM_  ; // secondary name to be given to HF collection of hits
      std::string ZDCRecHitCollectionDM_ ; // secondary name to be given to ZDC collection of hits
      // Digis:
      // output:
      std::string HBHEDigiCollectionDM_; // secondary name to be given to HBHE collection of hits
      std::string HODigiCollectionDM_  ; // secondary name to be given to HO collection of hits
      std::string HFDigiCollectionDM_  ; // secondary name to be given to HF collection of hits
      std::string ZDCDigiCollectionDM_ ; // secondary name to be given to ZDC collection of hits
      std::string QIE10DigiCollectionDM_ ; // secondary name to be given to QIE10 collection of hits
      std::string QIE11DigiCollectionDM_ ; // secondary name to be given to QIE11 collection of hits

      // Muons
      // output:
      std::string DTDigiCollectionDM_;       // secondary name to be given to new DT digis
      std::string RPCDigiCollectionDM_;      // secondary name to be given to new RPC digis
      std::string CSCStripDigiCollectionDM_; // secondary name given to new collection of CSC Strip digis
      std::string CSCWireDigiCollectionDM_;  // secondary name given to new collection of CSC wire digis
      std::string CSCComparatorDigiCollectionDM_; // "     "                              CSC Comparator digis

      // SiStrips
      std::string SiStripDigiCollectionDM_  ; // secondary name to be given to new SiStrip digis

      // SiPixels
      std::string PixelDigiCollectionDM_  ; // secondary name to be given to new SiPixel digis

      // merge tracker digis or tracks?
      bool MergeTrackerDigis_;

      // Submodules to handle the individual detectors

      DataMixingPileupCopy  *PUWorker_;

      DataMixingEMWorker *EMWorker_ ;
      DataMixingEMDigiWorker *EMDigiWorker_ ;
      bool MergeEMDigis_;

      // Hcal 
      
      DataMixingHcalWorker *HcalWorker_ ;
      DataMixingHcalDigiWorker *HcalDigiWorker_ ;
      DataMixingHcalDigiWorkerProd *HcalDigiWorkerProd_ ;

     // tokens needed to DataMixingHcalDigiWorkerProd
      edm::InputTag EBPileInputTag_; // InputTag for Pileup Digis collection  
      edm::InputTag EEPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag ESPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag HBHEPileInputTag_; // InputTag for Pileup Digis collection  
      edm::InputTag HOPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag HFPileInputTag_  ; // InputTag for Pileup Digis collection
      edm::InputTag ZDCPileInputTag_ ; // InputTag for Pileup Digis collection
      edm::InputTag QIE10PileInputTag_ ; // InputTag for Pileup Digis collection
      edm::InputTag QIE11PileInputTag_ ; // InputTag for Pileup Digis collection
      edm::EDGetTokenT<HBHEDigitizerTraits::DigiCollection> tok_hbhe_;
      edm::EDGetTokenT<HODigitizerTraits::DigiCollection> tok_ho_;
      edm::EDGetTokenT<HFDigitizerTraits::DigiCollection> tok_hf_;
      edm::EDGetTokenT<ZDCDigitizerTraits::DigiCollection> tok_zdc_;
      edm::EDGetTokenT<HcalQIE10DigitizerTraits::DigiCollection> tok_qie10_;
      edm::EDGetTokenT<HcalQIE11DigitizerTraits::DigiCollection> tok_qie11_;

      bool MergeHcalDigis_;
      bool MergeHcalDigisProd_;

      bool MergePileup_;
      bool AddedPileup_;

      // Muons

      DataMixingMuonWorker *MuonWorker_ ;

      // Si-Strips

      DataMixingSiStripWorker *SiStripWorker_ ;
      DataMixingSiStripRawWorker *SiStripRawWorker_ ;
      bool useSiStripRawDigi_;
      std::string siStripRawDigiSource_;

      // Pixels

      DataMixingSiPixelWorker *SiPixelWorker_ ;


      virtual void getSubdetectorNames();  

      // internally used information : subdetectors present in input
      std::vector<std::string> Subdetectors_;

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

    };
}//edm

#endif
