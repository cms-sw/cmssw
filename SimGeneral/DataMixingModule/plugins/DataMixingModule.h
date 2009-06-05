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
      edm::InputTag EBrechitCollection_; // secondary name given to collection of EB rechits
      edm::InputTag EErechitCollection_; // secondary name given to collection of EE rechits
      edm::InputTag ESrechitCollection_; // secondary name given to collection of EE rechits
      std::string EBRecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string EERecHitCollectionDM_; // secondary name to be given to EE collection of hits
      std::string ESRecHitCollectionDM_; // secondary name to be given to EE collection of hits

      // Hcal
      edm::InputTag HBHErechitCollection_; // secondary name given to collection of EB rechits
      edm::InputTag HOrechitCollection_  ; // secondary name given to collection of EB rechits
      edm::InputTag HFrechitCollection_  ; // secondary name given to collection of EB rechits
      edm::InputTag ZDCrechitCollection_ ; // secondary name given to collection of EB rechits
      std::string HBHERecHitCollectionDM_; // secondary name to be given to EB collection of hits
      std::string HORecHitCollectionDM_  ; // secondary name to be given to EB collection of hits
      std::string HFRecHitCollectionDM_  ; // secondary name to be given to EB collection of hits
      std::string ZDCRecHitCollectionDM_ ; // secondary name to be given to EB collection of hits

      // Muons
      edm::InputTag DTdigi_collection_;      // secondary name given to collection of DT digis
      edm::InputTag RPCdigi_collection_;     // secondary name given to collection of RPC digis
      edm::InputTag CSCstripdigi_collection_;// secondary name given to collection of CSC Strip digis
      edm::InputTag CSCwiredigi_collection_; // secondary name given to collection of CSC wire digis

      std::string DTDigiCollectionDM_;       // secondary name to be given to new DT digis
      std::string RPCDigiCollectionDM_;      // secondary name to be given to new RPC digis
      std::string CSCStripDigiCollectionDM_; // secondary name given to new collection of CSC Strip digis
      std::string CSCWireDigiCollectionDM_;  // secondary name given to new collection of CSC wire digis

      // SiStrips
      edm::InputTag Sistripdigi_collection_ ; // secondary name given to collection of SiStrip digis
      std::string SiStripDigiCollectionDM_  ; // secondary name to be given to new SiStrip digis

      // SiPixels
      edm::InputTag pixeldigi_collection_ ; // secondary name given to collection of SiPixel digis
      std::string PixelDigiCollectionDM_  ; // secondary name to be given to new SiPixel digis

      // Submodules to handle the individual detectors

      DataMixingEMWorker *EMWorker_ ;

      // Hcal 
      
      DataMixingHcalWorker *HcalWorker_ ;

      // Muons

      DataMixingMuonWorker *MuonWorker_ ;

      // Si-Strips

      DataMixingSiStripWorker *SiStripWorker_ ;

      // Pixels

      DataMixingSiPixelWorker *SiPixelWorker_ ;


      virtual void put(edm::Event &e) ;
      virtual void createnewEDProduct();
      virtual void addSignals(const edm::Event &e); 
      virtual void doPileUp(edm::Event &e);
      virtual void addPileups(const int bcr, edm::Event*,unsigned int EventId,unsigned int worker);
      virtual void getSubdetectorNames();

      // internally used information : subdetectors present in input
      std::vector<std::string> Subdetectors_;

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
