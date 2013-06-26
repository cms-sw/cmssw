// This is CSCBaseElectronicsSim.cc

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCBaseElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include<list>
#include<algorithm>

CSCBaseElectronicsSim::CSCBaseElectronicsSim(const edm::ParameterSet & p)
: 
  theSpecs(0),
  theLayerGeometry(0),
  theLayer(0),
  theSignalMap(),
  theAmpResponse(),
  theBunchSpacing(25.),
  theNoiseWasAdded(false),
  nElements(0),
  theShapingTime(p.getParameter<int>("shapingTime")),
  thePeakTimeSigma(p.getParameter<double>("peakTimeSigma")),
  theBunchTimingOffsets(p.getParameter<std::vector<double> >("bunchTimingOffsets")),
  theSignalStartTime(p.getParameter<double>("signalStartTime")),
  theSignalStopTime(p.getParameter<double>("signalStopTime")),
  theSamplingTime(p.getParameter<double>("samplingTime")),
  theNumberOfSamples(static_cast<int>((theSignalStopTime-theSignalStartTime)/theSamplingTime)),
  theOffsetOfBxZero(p.getParameter<int>("timeBitForBxZero")),
  theSignalPropagationSpeed(p.getParameter<std::vector<double> >("signalSpeed")),
  theTimingCalibrationError(p.getParameter<std::vector<double> >("timingCalibrationError")),
  doNoise_(p.getParameter<bool>("doNoise")),
  theRandGaussQ(0)
{
  assert(theBunchTimingOffsets.size() == 11);
}


CSCBaseElectronicsSim::~CSCBaseElectronicsSim()
{
  delete theRandGaussQ;
}


void CSCBaseElectronicsSim::setRandomEngine(CLHEP::HepRandomEngine& engine)
{
  if(theRandGaussQ) delete theRandGaussQ;
  theRandGaussQ = new CLHEP::RandGaussQ(engine);
}


void CSCBaseElectronicsSim::simulate(const CSCLayer * layer,
                               const std::vector<CSCDetectorHit> & detectorHits)
{
  theNoiseWasAdded = false;

  {
    theSignalMap.clear();
    theDetectorHitMap.clear();
    setLayer(layer);
    // can we swap for efficiency?
    theDigiSimLinks = DigiSimLinks(layerId().rawId());
  }
  
  {
    size_t nHits = detectorHits.size();
      // turn each detector hit into an analog signal
    for( size_t i = 0; i < nHits; ++i) {
      int element = readoutElement( detectorHits[i].getElement() );

      // skip if  hit element is not part of a readout element
      // e.g. wire in non-readout group
      if ( element != 0 ) add( amplifySignal(detectorHits[i]) );
    }
  }
  
  {
    if(doNoise_) {
      addNoise();
    }
  }
} 


void CSCBaseElectronicsSim::setLayer(const CSCLayer * layer) 
{
  // fill the specs member data
  theSpecs = layer->chamber()->specs();
  theLayerGeometry = layer->geometry();

  theLayer = layer;
  theLayerId = CSCDetId(theLayer->geographicalId().rawId());
  initParameters();
}


void CSCBaseElectronicsSim::fillAmpResponse() {
  std::vector<float> ampBinValues(theNumberOfSamples);
  int i = 0;
  for( ; i < theNumberOfSamples; ++i) {
    ampBinValues[i] = calculateAmpResponse(i*theSamplingTime);
    // truncate any entries that are trivially small
    if(i>5 && ampBinValues[i] < 0.000001) break;
  }
  ampBinValues.resize(i);
  theAmpResponse = CSCAnalogSignal(0, theSamplingTime, ampBinValues, 1., 0.);

  LogTrace("CSCBaseElectronicsSim") << 
    "CSCBaseElectronicsSim: dump of theAmpResponse follows...\n"
	 << theAmpResponse;
}



CSCAnalogSignal
CSCBaseElectronicsSim::amplifySignal(const CSCDetectorHit & detectorHit)  {
  int element = readoutElement(detectorHit.getElement());

  float readoutTime = detectorHit.getTime() 
                    + signalDelay(element, detectorHit.getPosition());

  // start from the amp response, and modify it.
  CSCAnalogSignal thisSignal(theAmpResponse);
  thisSignal *= detectorHit.getCharge();
  thisSignal.setTimeOffset(readoutTime);
  thisSignal.setElement(element);
  // keep track of links between digis and hits
  theDetectorHitMap.insert( DetectorHitMap::value_type(channelIndex(element), detectorHit) );
  return thisSignal;
} 


CSCAnalogSignal CSCBaseElectronicsSim::makeNoiseSignal(int element) {
  std::vector<float> binValues(theNumberOfSamples);
  // default is empty
  return CSCAnalogSignal(element, theSamplingTime, binValues, 0., theSignalStartTime);
} 


void CSCBaseElectronicsSim::addNoise() {
  for(CSCSignalMap::iterator mapI = theSignalMap.begin(); 
      mapI!=  theSignalMap.end(); ++mapI) {
    // superimpose electronics noise
    (*mapI).second.superimpose(makeNoiseSignal((*mapI).first));
    // DON'T do amp gain variations.  Handled in strips by calibration code
    // and variations in the shaper peaking time.
     double timeOffset = theRandGaussQ->fire((*mapI).second.getTimeOffset(), thePeakTimeSigma);
    (*mapI).second.setTimeOffset(timeOffset);
  }
  theNoiseWasAdded = true;
}


CSCAnalogSignal & CSCBaseElectronicsSim::find(int element) {
  if(element <= 0 || element > nElements) {
    LogTrace("CSCBaseElectronicsSim") << "CSCBaseElectronicsSim: bad element = " << element << 
         ". There are " << nElements  << " elements.";
    edm::LogError("Error in CSCBaseElectronicsSim:  element out of bounds");
  }
  CSCSignalMap::iterator signalMapItr = theSignalMap.find(element);
  if(signalMapItr == theSignalMap.end()) {
    CSCAnalogSignal newSignal;
    if(theNoiseWasAdded) {
      newSignal = makeNoiseSignal(element);
    } else {
      std::vector<float> emptyV(theNumberOfSamples);
      newSignal = CSCAnalogSignal(element, theSamplingTime, emptyV, 0., theSignalStartTime);
    }
    signalMapItr = theSignalMap.insert( std::pair<int, CSCAnalogSignal>(element, newSignal) ).first;
  }
  return (*signalMapItr).second;
}


CSCAnalogSignal & CSCBaseElectronicsSim::add(const CSCAnalogSignal & signal) {
  int element = signal.getElement();
  CSCAnalogSignal & newSignal = find(element);
  newSignal.superimpose(signal);
  return newSignal;
}
 

float CSCBaseElectronicsSim::signalDelay(int element, float pos) const {
  // readout is on top edge of chamber for strips, right edge
  // for wires.
  // zero calibrated to chamber center
  float distance = -1. * pos;
  float speed = theSignalPropagationSpeed[theSpecs->chamberType()];
  return distance / speed;
}


void CSCBaseElectronicsSim::addLinks(int channelIndex) {
  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr 
    = theDetectorHitMap.equal_range(channelIndex);

  // find the fraction contribution for each SimTrack
  std::map<int,float> simTrackChargeMap;
  std::map<int, EncodedEventId> eventIdMap;
  float totalCharge = 0;
  for( DetectorHitMap::iterator hitItr = channelHitItr.first; 
                                hitItr != channelHitItr.second; ++hitItr){
    const PSimHit * hit = hitItr->second.getSimHit();
    // might be zero for unit tests and such
    if(hit != 0) {
      int simTrackId = hitItr->second.getSimHit()->trackId();
      float charge = hitItr->second.getCharge();
      std::map<int,float>::iterator chargeItr = simTrackChargeMap.find(simTrackId);
      if( chargeItr == simTrackChargeMap.end() ) {
        simTrackChargeMap[simTrackId] = charge;
        eventIdMap[simTrackId] = hit->eventId();
      } else {
        chargeItr->second += charge;
      }
      totalCharge += charge;
    }
  }

  for(std::map<int,float>::iterator chargeItr = simTrackChargeMap.begin(); 
                          chargeItr != simTrackChargeMap.end(); ++chargeItr) {
    int simTrackId = chargeItr->first;
    theDigiSimLinks.push_back( StripDigiSimLink(channelIndex, simTrackId,  
                                  eventIdMap[simTrackId], chargeItr->second/totalCharge ) );
    
  }
}



