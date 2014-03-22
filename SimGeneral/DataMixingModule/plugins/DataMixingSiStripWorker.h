#ifndef DataMixingSiStripWorker_h
#define SimDataMixingSiStripWorker_h

/** \class DataMixingSiStripWorker
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

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ModuleCallingContext;

  class DataMixingSiStripWorker
    {
    public:

      DataMixingSiStripWorker();

     /** standard constructor*/
      explicit DataMixingSiStripWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingSiStripWorker();

      void putSiStrip(edm::Event &e) ;
      void addSiStripSignals(const edm::Event &e); 
      void addSiStripPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                             ModuleCallingContext const*);


    private:
      // data specifiers

      edm::InputTag SistripLabelSig_ ;        // name given to collection of SiStrip digis
      edm::InputTag SiStripPileInputTag_ ;    // InputTag for pileup strips
      std::string SiStripDigiCollectionDM_  ; // secondary name to be given to new SiStrip digis

      edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > SiStripDigiToken_ ;  // Token to retrieve information            
      edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > SiStripDigiPToken_ ;  // Token to retrieve information           


      // 

      typedef std::vector<SiStripDigi> OneDetectorMap;   // maps by strip ID for later combination - can have duplicate strips
      typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex; // map to all data for each detector ID

      SiGlobalIndex SiHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

      class StrictWeakOrdering{
      public:
	bool operator() (SiStripDigi i,SiStripDigi j) const {return i.strip() < j.strip();}
      };


    };
}//edm

#endif
