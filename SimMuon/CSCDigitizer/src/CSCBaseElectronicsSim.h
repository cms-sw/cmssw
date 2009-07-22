#ifndef MU_END_BASE_ELECTRONICS_SIM_H
#define MU_END_BASE_ELECTRONICS_SIM_H

/** \class CSCBaseElectronicsSim
 *
 * Commonalities between  CSCStripElectronicsSim and CSCWireElectronicsSim.
 *
 * \author Rick Wilkinson
 *
 * It has three non-virtual functions, so that's enough
 * to deserve a new class.
 * And since it has virtual functions it needs a virtual dtor.
 *
 */

#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <map>

#include "DataFormats/Common/interface/DetSet.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "CLHEP/Random/RandGaussQ.h"

// declarations
class CSCLayer;
class CSCChamberSpecs;
class CSCDetectorHit;
class CSCLayerGeometry;
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
class DetId;
class PSimHit;

class CSCBaseElectronicsSim
{
public:

  typedef std::map<int, CSCAnalogSignal, std::less<int> > CSCSignalMap;
  typedef edm::DetSet<StripDigiSimLink> DigiSimLinks;
 
  void setRandomEngine(CLHEP::HepRandomEngine& engine);
 
  // takes the input detector hits, turns them into DIGIs, and
  // stores them in the layer
  void simulate(const CSCLayer * layer,
                const std::vector<CSCDetectorHit> & inputHits);

  const DigiSimLinks & digiSimLinks() const {return theDigiSimLinks;}
 
  virtual ~CSCBaseElectronicsSim();

  /// for standalone apps who don't calculate it from the geometry
  void setLayerId(const CSCDetId & id) {theLayerId = id;}

protected:
  // constructor protected, so this class must be a base class
  CSCBaseElectronicsSim(const edm::ParameterSet & p);

  void setLayer(const CSCLayer * layer);
  // initialize things that change from layer to layer
  virtual void initParameters() = 0;

  void fillAmpResponse();
  virtual float calculateAmpResponse(float t) const = 0;

  // this one turns CSCDetectorHits into CSCAnalogSignals
  CSCAnalogSignal amplifySignal(const CSCDetectorHit &);

  // returns readout element.  So wire 20 might be part of wire group 2.
  virtual int readoutElement(int element) const = 0; 

  //  fills the member map of signals on strips, superimposing any duplicates
  void combineAnalogSignals(const std::vector<CSCAnalogSignal> &);

  void setNoise(float rmsNoise, float noiseSigmaThreshold);

  /// How long before & after the bunch crossing to simulate
  /// shortening the time can save CPU
  void setSignalTimeRange(double startTime, double stopTime) {
    theSignalStartTime = startTime;
    theSignalStopTime = stopTime;
  }

  void addNoise();

  CSCAnalogSignal & find(int element);
  // the returned signal will be the one stored in the
  // signal, not the original.  If another signal
  // is found on this element, they will be superimposed.
  CSCAnalogSignal & add(const CSCAnalogSignal &);
  virtual CSCAnalogSignal makeNoiseSignal(int element);

  /// how long, in ns, it takes a signal at pos to propagate to
  /// the readout edge.  This may be negative, since the timing
  /// may be calibrated to the center of the detector
  virtual float signalDelay(int element, float pos) const = 0;

  /// creates links from Digi to SimTrack
  /// disabled for now
  virtual void addLinks(int channelIndex);

  /// lets users map channels to different indices for links
  virtual int channelIndex(int channel) const {return channel;}

  /// the CSCDetId corresponding to the current layer
  CSCDetId layerId() const {return theLayerId;}

  /// the average time-of-flight from the interaction point to the given detector
  double averageTimeOfFlight(const DetId & detId) const;

  // member data
  enum {NONE, CONSERVATIVE, RADICAL}; 

  const CSCChamberSpecs * theSpecs;
  const CSCLayerGeometry * theLayerGeometry;
  const CSCLayer * theLayer;  // the one currently being digitized
  CSCDetId theLayerId;

  CSCSignalMap theSignalMap;
  CSCAnalogSignal theAmpResponse;

  // Useful parameters
  float theBunchSpacing;

  // lets routines know whether new signals should
  //  have noise added, or just be empty.  If the
  //  noise hasn't been added yet, just make empty.
  bool theNoiseWasAdded;

  // the numbers of wire groups or strips in this layer
  int nElements;

  // amplifier parameters
  int theShapingTime;
  float thePeakTimeSigma;
  // used to correct the bunch timing so that the signal event 
  // comes at BX zero.
  std::vector<double> theBunchTimingOffsets;

  // when the signal is to be simulated
  float theSignalStartTime;
  float theSignalStopTime;

  // size of time bins for signal shape, in ns
  float theSamplingTime;

  // time bins for pulse shape
  int theNumberOfSamples;

  // Which bit in the 16-bit time word corresponds to the zeroth beam crossing?
  int theOffsetOfBxZero; // bit corresponding to bx 0 (counting from 0-15)

  bool doNoise_;

  // keeps track of which hits contribute to which channels
  typedef std::multimap<int, CSCDetectorHit, std::less<int> >  DetectorHitMap;
  DetectorHitMap theDetectorHitMap;
  DigiSimLinks theDigiSimLinks;

  CLHEP::RandGaussQ * theRandGaussQ;
};

#endif
