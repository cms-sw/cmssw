// This is CSCBaseElectronicsSim.cc

#include "Utilities/Timing/interface/TimingReport.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/CSCDigitizer/src/CSCBaseElectronicsSim.h"
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"
#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "CLHEP/Random/RandGaussQ.h" 
#include "CLHEP/Units/PhysicalConstants.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include<vector>
#include<map>
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
  theTailShaping(RADICAL),
  theAmpGainVariance(p.getParameter<double>("ampGainVariance")),
  thePeakTimeVariance(p.getParameter<double>("peakTimeVariance")),
  theBunchTimingOffsets(p.getParameter<std::vector<double> >("bunchTimingOffsets")),
  theSignalStartTime(p.getParameter<double>("signalStartTime")),
  theSignalStopTime(p.getParameter<double>("signalStopTime")),
  theSamplingTime(p.getParameter<double>("samplingTime")),
  theNumberOfSamples(static_cast<int>((theSignalStopTime-theSignalStartTime)/theSamplingTime)),
  theOffsetOfBxZero(p.getParameter<int>("timeBitForBxZero")),
  doNoise_(p.getParameter<bool>("doNoise"))
{
  assert(theBunchTimingOffsets.size() == 11);
}


void CSCBaseElectronicsSim::simulate(const CSCLayer * layer,
                               const std::vector<CSCDetectorHit> & detectorHits)
{
  theNoiseWasAdded = false;

  TimeMe a("CSCBaseEl:simulate");
  {
    theSignalMap.clear();
    //theDetectorHitMap.clear();
    // fill the specs member data
    theSpecs = layer->chamber()->specs();
    theLayer = layer;
    initParameters();
  }
  
  {
    TimeMe c("CSCBaseEl:loop");
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

  LogDebug("CSCBaseElectronicsSim") << 
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
  //theDetectorHitMap.insert( DetectorHitMap::value_type(channelIndex(element), detectorHit) );
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
    // do amp gain variations
    (*mapI).second *= (1.+ theAmpGainVariance * RandGaussQ::shoot());
    // and variations in the shaper peaking time.
    (*mapI).second.setTimeOffset((*mapI).second.getTimeOffset() 
                               + thePeakTimeVariance * RandGaussQ::shoot());
  }
  theNoiseWasAdded = true;
}


CSCAnalogSignal & CSCBaseElectronicsSim::find(int element) {
  if(element <= 0 || element > nElements) {
    LogDebug("CSCBaseElectronicsSim") << "MEBES: bad element = " << element << 
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
 
/*
void CSCBaseElectronicsSim::addLinks(int channelIndex) {
  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr 
    = theDetectorHitMap.equal_range(channelIndex);

  // find the fraction contribution for each SimTrack
  std::map<int,float> simTrackChargeMap;
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
        simTrackChargeMap.insert( std::pair<int,float>(simTrackId, charge) );
      } else {
        chargeItr->second += charge;
      }
      totalCharge += charge;
    }
  }

  for(std::map<int,float>::iterator chargeItr = simTrackChargeMap.begin(); 
                          chargeItr != simTrackChargeMap.end(); ++chargeItr) {
    theLayer->simDet()->addLink( channelIndex, chargeItr->first, chargeItr->second/totalCharge);
  }
}

*/

CSCDetId CSCBaseElectronicsSim::layerId() const {
  return CSCDetId(theLayer->geographicalId().rawId());
}


