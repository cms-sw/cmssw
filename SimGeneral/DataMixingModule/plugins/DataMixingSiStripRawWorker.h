#ifndef DataMixingSiStripRawWorker_h
#define SimDataMixingSiStripRawWorker_h

/** \class DataMixingSiStripRawWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the SiStrip information
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
//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ModuleCallingContext;

  class DataMixingSiStripRawWorker
    {
    public:

      DataMixingSiStripRawWorker();

     /** standard constructor*/
      explicit DataMixingSiStripRawWorker(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC);

      /**Default destructor*/
      virtual ~DataMixingSiStripRawWorker();

      void putSiStrip(edm::Event &e) ;
      void addSiStripSignals(const edm::Event &e); 
      void addSiStripPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                             ModuleCallingContext const*);


    private:
      // data specifiers

      edm::InputTag Sistripdigi_collectionSig_ ; // primary name given to collection of SiStrip digis
      edm::InputTag SistripLabelSig_ ;           // secondary name given to collection of SiStrip digis
      edm::InputTag SiStripPileInputTag_;        // InputTag for pileup strips
      edm::InputTag SiStripRawInputTag_;         // InputTag for strips with rawdigis
      edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>> SiStripInputTok_ ; // InputToken for strips with rawdigis
      edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi>> SiStripRawInputTok_ ; // InputToken for strips with rawdigis
      std::string SiStripDigiCollectionDM_  ;    // secondary name to be given to new SiStrip raw digis
      std::string SiStripRawDigiSource_ ;        // which collection is rawdigis: either "SIGNAL" or "PILEUP" 

      // 

      typedef std::vector<SiStripDigi> OneDetectorMap;   // maps by strip ID for later combination - can have duplicate strips
      typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex; // map to all data for each detector ID
      
      SiGlobalIndex SiHitStorage_;

      // SiStripDigi and SiStripRawDigi collections
      const edm::DetSetVector<SiStripDigi>    *digicollection_;
      const edm::DetSetVector<SiStripRawDigi> *rawdigicollection_;

      std::string label_;

    };
}//edm

#endif
