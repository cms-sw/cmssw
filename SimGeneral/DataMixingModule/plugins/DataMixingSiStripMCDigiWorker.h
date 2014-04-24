#ifndef DataMixingSiStripMCDigiWorker_h
#define SimDataMixingSiStripMCDigiWorker_h

/** \class DataMixingSiStripMCDigiWorker
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

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


#include <map>
#include <vector>
#include <string>

namespace edm
{
  class ModuleCallingContext;
  class ConsumesCollector;
 
  class DataMixingSiStripMCDigiWorker
    {
    public:

      DataMixingSiStripMCDigiWorker();

     /** standard constructor*/
      explicit DataMixingSiStripMCDigiWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingSiStripMCDigiWorker();

      void putSiStrip(edm::Event &e, edm::EventSetup const& iSetup) ;
      void addSiStripSignals(const edm::Event &e); 
      void addSiStripPileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                             ModuleCallingContext const*);


      virtual void initializeEvent(const edm::Event &e, edm::EventSetup const& iSetup);

      void DMinitializeDetUnit(StripGeomDetUnit* det, const edm::EventSetup& iSetup );

    private:
      // data specifiers

      edm::InputTag SistripLabelSig_ ;        // name given to collection of SiStrip digis
      edm::InputTag SiStripPileInputTag_ ;    // InputTag for pileup strips
      std::string SiStripDigiCollectionDM_  ; // secondary name to be given to new SiStrip digis

      // 

      typedef std::vector<SiStripDigi> OneDetectorMap;   // maps by strip ID for later combination - can have duplicate strips
      typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex; // map to all data for each detector ID
      typedef SiDigitalConverter::DigitalVecType DigitalVecType;

      SiGlobalIndex SiHitStorage_;


      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      std::string label_;

      std::string gainLabel;
      bool peakMode;
      double theThreshold;
      double theElectronPerADC;
      int theFedAlgo;
      std::string geometryType;

      std::unique_ptr<SiGaussianTailNoiseAdder> theSiNoiseAdder;
      std::unique_ptr<SiStripFedZeroSuppression> theSiZeroSuppress;
      std::unique_ptr<SiTrivialDigitalConverter> theSiDigitalConverter;


      edm::ESHandle<TrackerGeometry> pDD;

      // bad channels for each detector ID
      std::map<unsigned int, std::vector<bool> > allBadChannels;
      // first and last channel wit signal for each detector ID
      std::map<unsigned int, size_t> firstChannelsWithSignal;
      std::map<unsigned int, size_t> lastChannelsWithSignal;


      class StrictWeakOrdering{
      public:
	bool operator() (SiStripDigi i,SiStripDigi j) const {return i.strip() < j.strip();}
      };


    };
}//edm

#endif
