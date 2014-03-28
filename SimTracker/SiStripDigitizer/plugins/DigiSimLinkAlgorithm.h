#ifndef DigiSimLinkAlgorithm_h
#define DigiSimLinkAlgorithm_h

/** \DigiSimLinkAlgorithm
 *
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
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
#include "DigiSimLinkPileUpSignals.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

namespace CLHEP {
  class HepRandomEngine;
}

class TrackerTopolgoy;

class DigiSimLinkAlgorithm {
 public:
  typedef SiDigitalConverter::DigitalVecType DigitalVecType;
  typedef SiDigitalConverter::DigitalRawVecType DigitalRawVecType;
  typedef DigiSimLinkPileUpSignals::HitToDigisMapType HitToDigisMapType;
  typedef DigiSimLinkPileUpSignals::HitCounterToDigisMapType HitCounterToDigisMapType;
  typedef std::map< int, float, std::less<int> > hit_map_type;
  typedef float Amplitude;
  
  // Constructor
  DigiSimLinkAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);

  // Destructor
  ~DigiSimLinkAlgorithm();

  // Runs the algorithm
  void  run(edm::DetSet<SiStripDigi>&, edm::DetSet<SiStripRawDigi>&,
            const std::vector<std::pair<const PSimHit*, int > >  &, 
            StripGeomDetUnit *, GlobalVector, float, 
            edm::ESHandle<SiStripGain> &, edm::ESHandle<SiStripThreshold> &, 
            edm::ESHandle<SiStripNoises> &, edm::ESHandle<SiStripPedestals> &, edm::ESHandle<SiStripBadStrip> &,
	    const TrackerTopology *tTopo);

  // digisimlink
  std::vector<StripDigiSimLink> make_link() { return link_coll; }

  // ParticleDataTable
  void setParticleDataTable(const ParticleDataTable * pardt) {
  	theSiHitDigitizer->setParticleDataTable(pardt); 
  	pdt= pardt; 
  }
  
 private:
  edm::ParameterSet conf_;
  double theElectronPerADC;
  double theThreshold;
  double cmnRMStib;
  double cmnRMStob;
  double cmnRMStid;
  double cmnRMStec;
  double APVSaturationProb;          
  bool peakMode;
  bool noise;
  bool RealPedestals;              
  bool SingleStripNoise;          
  bool CommonModeNoise;           
  bool BaselineShift;             
  bool APVSaturationFromHIP;
  
  int theFedAlgo;
  bool zeroSuppression;
  double theTOFCutForPeak;
  double theTOFCutForDeconvolution;
  double tofCut;
  int numStrips; 
  int strip;     
  //double noiseRMS;
  //double pedValue;
  double cosmicShift;
  double inefficiency;
  double pedOffset;

  size_t firstChannelWithSignal;
  size_t lastChannelWithSignal;
  size_t localFirstChannel;
  size_t localLastChannel;

  // local amplitude of detector channels (from processed PSimHit)
  std::vector<float> locAmpl;
  // total amplitude of detector channels
  std::vector<float> detAmpl;

  const ParticleDataTable * pdt;
  const ParticleData * particle;
  
  SiHitDigitizer* theSiHitDigitizer;
  DigiSimLinkPileUpSignals* theDigiSimLinkPileUpSignals;
  SiGaussianTailNoiseAdder* theSiNoiseAdder;
  SiTrivialDigitalConverter* theSiDigitalConverter;
  SiStripFedZeroSuppression* theSiZeroSuppress;
  CLHEP::HepRandomEngine& rndEngine;

  DigitalVecType digis;
  DigitalRawVecType rawdigis;
  std::vector<StripDigiSimLink> link_coll;
  CLHEP::RandFlat *theFlatDistribution;

  void push_link(const DigitalVecType&,
		 const HitToDigisMapType&,
		 const HitCounterToDigisMapType&,
		 const std::vector<float>&,
		 unsigned int);
  
  void push_link_raw(const DigitalRawVecType&,
		     const HitToDigisMapType&,
		     const HitCounterToDigisMapType&,
		     const std::vector<float>&,
		     unsigned int);
};

#endif
