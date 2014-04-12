#ifndef DataMixingSiPixelWorker_h
#define SimDataMixingSiPixelWorker_h

/** \class DataMixingSiPixelWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the Si Pixel information
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
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include <map>
#include <vector>
#include <string>


namespace edm
{
  class ModuleCallingContext;

  class DataMixingSiPixelWorker
    {
    public:

      DataMixingSiPixelWorker();

     /** standard constructor*/
      explicit DataMixingSiPixelWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingSiPixelWorker();

      void putSiPixel(edm::Event &e) ;
      void addSiPixelSignals(const edm::Event &e); 
      void addSiPixelPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId, ModuleCallingContext const*);


    private:
      // data specifiers

      edm::InputTag pixeldigi_collectionSig_ ; // secondary name given to collection of SiPixel digis
      edm::InputTag pixeldigi_collectionPile_ ; // secondary name given to collection of SiPixel digis
      std::string PixelDigiCollectionDM_  ; // secondary name to be given to new SiPixel digis

      edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > PixelDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > PixelDigiPToken_ ;  // Token to retrieve information 

      // 

      typedef std::multimap<int, PixelDigi> OneDetectorMap;   // maps by pixel ID for later combination - can have duplicate pixels
      typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex; // map to all data for each detector ID

      SiGlobalIndex SiHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

    };
}//edm

#endif
