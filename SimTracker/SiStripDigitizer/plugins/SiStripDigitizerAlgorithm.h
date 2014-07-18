#ifndef SiStripDigitizerAlgorithm_h
#define SiStripDigitizerAlgorithm_h

/** \class SiStripDigitizerAlgorithm
 *
 * SiStripDigitizerAlgorithm converts hits to digis
 *
 ************************************************************/

#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "SiHitDigitizer.h"
#include "SimTracker/SiStripDigitizer/interface/SiPileUpSignals.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

class TrackerTopology;

namespace edm {
  class EventSetup;
}

class SiStripLorentzAngle;
class StripDigiSimLink;

namespace CLHEP {
  class HepRandomEngine;
}

class SiStripDigitizerAlgorithm {
 public:
  typedef SiDigitalConverter::DigitalVecType DigitalVecType;
  typedef SiDigitalConverter::DigitalRawVecType DigitalRawVecType;
  typedef SiPileUpSignals::SignalMapType SignalMapType;
  typedef std::map< int, float, std::less<int> > hit_map_type;
  typedef float Amplitude;
  
  // Constructor
  SiStripDigitizerAlgorithm(const edm::ParameterSet& conf);

  // Destructor
  ~SiStripDigitizerAlgorithm();

  void initializeDetUnit(StripGeomDetUnit* det, const edm::EventSetup& iSetup);

  void initializeEvent(const edm::EventSetup& iSetup);

  //run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         size_t inputBeginGlobalIndex,
                         const StripGeomDetUnit *stripdet,
                         const GlobalVector& bfield,
			 const TrackerTopology *tTopo,
                         CLHEP::HepRandomEngine*);

  void digitize(
                edm::DetSet<SiStripDigi>& outDigis,
                edm::DetSet<SiStripRawDigi>& outRawDigis,
                edm::DetSet<StripDigiSimLink>& outLink,
                const StripGeomDetUnit* stripdet,
                edm::ESHandle<SiStripGain>&,
                edm::ESHandle<SiStripThreshold>&, 
                edm::ESHandle<SiStripNoises>&,
                edm::ESHandle<SiStripPedestals>&,
                CLHEP::HepRandomEngine*);

  // ParticleDataTable
  void setParticleDataTable(const ParticleDataTable * pardt) {
  	theSiHitDigitizer->setParticleDataTable(pardt); 
  	pdt= pardt; 
  }
  
 private:
  const std::string lorentzAngleName;
  const double theThreshold;
  const double cmnRMStib;
  const double cmnRMStob;
  const double cmnRMStid;
  const double cmnRMStec;
  const double APVSaturationProb;          
  const bool makeDigiSimLinks_; //< Whether or not to create the association to sim truth collection. Set in configuration.
  const bool peakMode;
  const bool noise;
  const bool RealPedestals;              
  const bool SingleStripNoise;          
  const bool CommonModeNoise;           
  const bool BaselineShift;             
  const bool APVSaturationFromHIP;
  
  const int theFedAlgo;
  const bool zeroSuppression;
  const double theElectronPerADC;
  const double theTOFCutForPeak;
  const double theTOFCutForDeconvolution;
  const double tofCut;
  const double cosmicShift;
  const double inefficiency;
  const double pedOffset;

  const ParticleDataTable * pdt;
  const ParticleData * particle;
  
  const std::unique_ptr<SiHitDigitizer> theSiHitDigitizer;
  const std::unique_ptr<SiPileUpSignals> theSiPileUpSignals;
  const std::unique_ptr<const SiGaussianTailNoiseAdder> theSiNoiseAdder;
  const std::unique_ptr<SiTrivialDigitalConverter> theSiDigitalConverter;
  const std::unique_ptr<SiStripFedZeroSuppression> theSiZeroSuppress;

  // bad channels for each detector ID
  std::map<unsigned int, std::vector<bool> > allBadChannels;
  // first and last channel wit signal for each detector ID
  std::map<unsigned int, size_t> firstChannelsWithSignal;
  std::map<unsigned int, size_t> lastChannelsWithSignal;

  // ESHandles
  edm::ESHandle<SiStripLorentzAngle> lorentzAngleHandle;

  /** This structure is used to keep track of the SimTrack that contributed to each digi
      so that the truth association can be created.*/
  struct AssociationInfo
  {
    unsigned int trackID;
    EncodedEventId eventID;
    float contributionToADC;
    size_t simHitGlobalIndex; ///< The array index of the sim hit, but in the array for all crossings
  };

  typedef std::map<int, std::vector<AssociationInfo> >  AssociationInfoForChannel;
  typedef std::map<uint32_t, AssociationInfoForChannel>  AssociationInfoForDetId;
  /// Structure that holds the information on the SimTrack contributions. Only filled if makeDigiSimLinks_ is true.
  AssociationInfoForDetId associationInfoForDetId_;

};

#endif
