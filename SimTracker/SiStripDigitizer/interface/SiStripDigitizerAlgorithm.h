#ifndef SiStripDigitizerAlgorithm_h
#define SiStripDigitizerAlgorithm_h

/** \class SiStripDigitizerAlgorithm
 *
 * SiStripDigitizerAlgorithm converts hits to digis
 *
 * \author Andrea Giammanco
 *
 * \version   1st Version Sep. 29, 2005
 *
 ************************************************************/

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialDigitalConverter.h"
#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "SimTracker/SiStripDigitizer/interface/SiHitDigitizer.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripFedZeroSuppression.h"

namespace CLHEP {
  class HepRandomEngine;
}

class SiStripDigitizerAlgorithm 
{
 public:

  typedef SiDigitalConverter::DigitalVecType DigitalVecType;
  typedef SiDigitalConverter::DigitalRawVecType DigitalRawVecType;
  typedef SiPileUpSignals::signal_map_type signal_map_type;
  typedef SiPileUpSignals::HitToDigisMapType HitToDigisMapType;
  typedef std::map< int, float, std::less<int> > hit_map_type;
  typedef float Amplitude;

  //digisimlink
  std::vector<StripDigiSimLink> link_coll;
  std::vector<StripDigiSimLink> make_link(){ return link_coll;}

  
  SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);

  ~SiStripDigitizerAlgorithm();

  // Runs the algorithm
  void  run(edm::DetSet<SiStripDigi>&,edm::DetSet<SiStripRawDigi>&,const std::vector<PSimHit> &, StripGeomDetUnit *,GlobalVector,
	    float , edm::ESHandle<SiStripGain> &,edm::ESHandle<SiStripPedestals> &, edm::ESHandle<SiStripNoises> &);

  void setParticleDataTable(const ParticleDataTable * pdt);
  
 private:
  edm::ParameterSet conf_;
  //-- make_digis 
  float theElectronPerADC;     // Gain, number of electrons per adc count.
  float theThreshold;          
  bool peakMode;
  bool noise;
  int theFedAlgo;
  bool zeroSuppression;
  float theTOFCutForPeak;
  float theTOFCutForDeconvolution;
  float tofCut;                // Cut on the particle TOF


  SiHitDigitizer* theSiHitDigitizer;
  SiPileUpSignals* theSiPileUpSignals;
  SiGaussianTailNoiseAdder* theSiNoiseAdder;
  SiTrivialDigitalConverter* theSiDigitalConverter;
  SiStripFedZeroSuppression* theSiZeroSuppress;

  int numStrips; 
  int strip;     
  float noiseRMS;
  float cosmicShift;

  void push_link(const DigitalVecType&,
		 const HitToDigisMapType&,
		 const SiPileUpSignals::signal_map_type&,
		 unsigned int);
 
  void push_link_raw(const DigitalRawVecType&,
		     const HitToDigisMapType&,
		     const SiPileUpSignals::signal_map_type&,
		     unsigned int);
 
  CLHEP::HepRandomEngine& rndEngine;

  DigitalVecType digis;
  DigitalRawVecType rawdigis;
};

#endif
