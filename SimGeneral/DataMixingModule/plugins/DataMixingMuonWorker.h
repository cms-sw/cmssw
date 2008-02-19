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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Selector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

//DT
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
//RPC
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
//CSC
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class DataMixingMuonWorker
    {
    public:

      DataMixingMuonWorker();

     /** standard constructor*/
      explicit DataMixingMuonWorker(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~DataMixingMuonWorker();

      void putMuon(edm::Event &e) ;
      void addMuonSignals(const edm::Event &e); 
      void addMuonPileups(const int bcr, edm::Event*,unsigned int EventId);


    private:
      // data specifiers

      edm::InputTag DTdigi_collection_; // secondary name given to collection of DT digis
      edm::InputTag RPCdigi_collection_; // secondary name given to collection of RPC digis
      edm::InputTag CSCDigiTag_; // primary? name given to collection of CSC digis
      edm::InputTag DTDigiTag_; // primary? name given to collection of DT digis
      edm::InputTag RPCDigiTag_; // primary? name given to collection of RPC digis
      edm::InputTag CSCstripdigi_collection_; // secondary name given to collection of CSC Strip digis
      edm::InputTag CSCwiredigi_collection_; // secondary name given to collection of CSC wire digis

      std::string DTDigiCollectionDM_; // secondary name to be given to new DT digis
      std::string RPCDigiCollectionDM_; // secondary name to be given to new RPC digis
      std::string CSCStripDigiCollectionDM_; // secondary name given to new collection of CSC Strip digis
      std::string CSCWireDigiCollectionDM_; // secondary name given to new collection of CSC wire digis

      // just hold our own DigiCollections - order of digis in layer doesn't appear to matter...
      // will make a copy and put this back into the event...

      DTDigiCollection* OurDTDigis_;  
      RPCDigiCollection* OurRPCDigis_;  
      CSCStripDigiCollection* OurCSCStripDigis_;  
      CSCWireDigiCollection* OurCSCWireDigis_;  

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
