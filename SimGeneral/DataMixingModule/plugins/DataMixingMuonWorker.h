#ifndef DataMixingMuonWorker_h
#define SimDataMixingMuonWorker_h

/** \class DataMixingMuonWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the Muon information
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

//DT
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
//RPC
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
//CSC
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ModuleCallingContext;

  class DataMixingMuonWorker
    {
    public:

      DataMixingMuonWorker();

     /** standard constructor*/
      explicit DataMixingMuonWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingMuonWorker();

      void putMuon(edm::Event &e) ;
      void addMuonSignals(const edm::Event &e); 
      void addMuonPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId, ModuleCallingContext const*);


    private:
      // data specifiers

      edm::InputTag DTdigi_collectionSig_; // secondary name given to collection of DT digis
      edm::InputTag RPCdigi_collectionSig_; // secondary name given to collection of RPC digis
      edm::InputTag CSCDigiTagSig_; // primary? name given to collection of CSC digis
      edm::InputTag DTDigiTagSig_; // primary? name given to collection of DT digis
      edm::InputTag RPCDigiTagSig_; // primary? name given to collection of RPC digis
      edm::InputTag CSCstripdigi_collectionSig_; // secondary name given to collection of CSC Strip digis
      edm::InputTag CSCwiredigi_collectionSig_; // secondary name given to collection of CSC wire digis
      edm::InputTag CSCCompdigi_collectionSig_; // secondary name given to collection of CSC wire digis

      edm::InputTag DTPileInputTag_       ; // InputTag for pileup DTs
      edm::InputTag RPCPileInputTag_      ; // InputTag for pileup RPCs
      edm::InputTag CSCWirePileInputTag_  ; // InputTag for pileup CSC Wires
      edm::InputTag CSCStripPileInputTag_ ; // InputTag for pileup CSC Strips
      edm::InputTag CSCCompPileInputTag_ ; // InputTag for pileup CSC Comparators

      edm::EDGetTokenT<DTDigiCollection> DTDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCStripDigiCollection> CSCStripDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCWireDigiCollection> CSCWireDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCComparatorDigiCollection> CSCCompDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<RPCDigiCollection> RPCDigiToken_ ;  // Token to retrieve information 

      edm::EDGetTokenT<DTDigiCollection> DTDigiPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCStripDigiCollection> CSCStripDigiPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCWireDigiCollection> CSCWireDigiPToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<CSCComparatorDigiCollection> CSCCompDigiPToken_ ;  // Token to retrieve information
      edm::EDGetTokenT<RPCDigiCollection> RPCDigiPToken_ ;  // Token to retrieve information 

      std::string DTDigiCollectionDM_; // secondary name to be given to new DT digis
      std::string RPCDigiCollectionDM_; // secondary name to be given to new RPC digis
      std::string CSCStripDigiCollectionDM_; // secondary name given to new collection of CSC Strip digis
      std::string CSCWireDigiCollectionDM_; // secondary name given to new collection of CSC wire digis
      std::string CSCComparatorDigiCollectionDM_; // secondary name given to new collection of CSC comparator digis

      // just hold our own DigiCollections - order of digis in layer doesn't appear to matter...
      // will make a copy and put this back into the event...

      DTDigiCollection* OurDTDigis_;  
      RPCDigiCollection* OurRPCDigis_;  
      CSCStripDigiCollection* OurCSCStripDigis_;  
      CSCWireDigiCollection* OurCSCWireDigis_;  
      CSCComparatorDigiCollection* OurCSCComparatorDigis_;  

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

    };
}//edm

#endif
