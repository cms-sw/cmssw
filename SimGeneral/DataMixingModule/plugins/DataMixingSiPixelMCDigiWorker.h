#ifndef DataMixingSiPixelMCDigiWorker_h
#define DataMixingSiPixelMCDigiWorker_h

/** \class DataMixingSiPixelMCDigiWorker
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
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <map>
#include <vector>
#include <string>


namespace CLHEP {
  class HepRandomEngine;
}


namespace edm
{
  class ModuleCallingContext;

  class DataMixingSiPixelMCDigiWorker
    {
    public:

      DataMixingSiPixelMCDigiWorker();

     /** standard constructor*/
      explicit DataMixingSiPixelMCDigiWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingSiPixelMCDigiWorker();

      virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c); // override?

      void putSiPixel(edm::Event &e, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> &ps, int &bs) ;
      void addSiPixelSignals(const edm::Event &e); 
      void addSiPixelPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId, ModuleCallingContext const*);

      void setPileupInfo(const std::vector<PileupSummaryInfo> &ps, const int &bs); //this sets pu_scale


    private:

      //
      // PixelEfficiencies struct
      //
      /**
       * Internal use only.
       */
      struct PixelEfficiencies {
	PixelEfficiencies(const edm::ParameterSet& conf, bool AddPixelInefficiency, int NumberOfBarrelLayers, int NumberOfEndcapDisks);
	double thePixelEfficiency[20];     // Single pixel effciency
	double thePixelColEfficiency[20];  // Column effciency
	double thePixelChipEfficiency[20]; // ROC efficiency
	std::vector<double> theLadderEfficiency_BPix[20]; // Ladder efficiency
	std::vector<double> theModuleEfficiency_BPix[20]; // Module efficiency
	std::vector<double> thePUEfficiency[20]; // Instlumi dependent efficiency
	double theInnerEfficiency_FPix[20]; // Fpix inner module efficiency
	double theOuterEfficiency_FPix[20]; // Fpix outer module efficiency
	unsigned int FPixIndex;         // The Efficiency index for FPix Disks
      };

      // Needed by dynamic inefficiency 
      // 0-3 BPix, 4-5 FPix (inner, outer)
      double _pu_scale[20];

      // data specifiers

      edm::InputTag pixeldigi_collectionSig_ ; // secondary name given to collection of SiPixel digis
      edm::InputTag pixeldigi_collectionPile_ ; // secondary name given to collection of SiPixel digis
      std::string PixelDigiCollectionDM_  ; // secondary name to be given to new SiPixel digis

      edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > PixelDigiToken_ ;  // Token to retrieve information 
      edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > PixelDigiPToken_ ;  // Token to retrieve information 

      edm::ESHandle<TrackerGeometry> pDD;

      // 
      // Internal typedefs

      typedef int Amplitude;
      typedef std::map<int, Amplitude, std::less<int> > signal_map_type;  // from Digi.Skel.
      typedef signal_map_type::iterator          signal_map_iterator; // from Digi.Skel.  
      typedef signal_map_type::const_iterator    signal_map_const_iterator; // from Digi.Skel.  
      typedef std::map<uint32_t, signal_map_type> signalMaps;
      
      // Contains the accumulated hit info.
      signalMaps _signal;

      typedef std::multimap<int, PixelDigi> OneDetectorMap;   // maps by pixel ID for later combination - can have duplicate pixels
      typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex; // map to all data for each detector ID

      SiGlobalIndex SiHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;
      const std::string geometryType_;

      //-- Allow for upgrades        
      const int NumberOfBarrelLayers;     // Default = 3  
      const int NumberOfEndcapDisks;      // Default = 2  

      const double theInstLumiScaleFactor;
      const double bunchScaleAt25;

      const bool AddPixelInefficiency;        // bool to read in inefficiencies    

      const PixelEfficiencies pixelEff_;


    };
}//edm

#endif
