using namespace std;
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h" 
#include "SimDataFormats/CaloHit/interface/PCaloHit.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include<iostream>

using namespace cms;

CaloHitResponse::CaloHitResponse(const CaloVSimParameterMap * parametersMap, 
                                 const CaloVShape * shape)
: theParameterMap(parametersMap), 
  theShape(shape),  
  theHitCorrection(0),
  theGeometry(0),
  theMinBunch(-10), 
  theMaxBunch(10)
{
}


void CaloHitResponse::setBunchRange(int minBunch, int maxBunch) {
  theMinBunch = minBunch;
  theMaxBunch = maxBunch;
}


void CaloHitResponse::run(const vector<PCaloHit> & hits) {
  for(vector<PCaloHit>::const_iterator hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr)
  {
    LogDebug("CaloHitResponse") << *hitItr;
    CaloSamples signal = makeAnalogSignal(*hitItr);
    LogDebug("CaloHitResponse") << signal;
    // if there's already a frame for this in the map, superimpose it
    const DetId id(hitItr->id());
    map<DetId, CaloSamples>::iterator mapItr = theAnalogSignalMap.find(id);
    if (mapItr == theAnalogSignalMap.end()) {
      theAnalogSignalMap.insert(pair<DetId, CaloSamples>(id, signal));
    } else  {
      // need a "+=" to CaloSamples
      int sampleSize =  mapItr->second.size();
      assert(sampleSize == signal.size());
      assert(signal.presamples() == mapItr->second.presamples());

      for(int i = 0; i < sampleSize; ++i) {
        (mapItr->second)[i] += signal[i];
      }
    }
  }
}


CaloSamples CaloHitResponse::makeAnalogSignal(const PCaloHit & inputHit) const {

  // see if we need to correct the hit 
  PCaloHit hit = inputHit;
  if(theHitCorrection != 0) {
    theHitCorrection->correct(hit);
  }

  DetId detId(hit.id());
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);


  double signal = analogSignalAmplitude(hit, parameters);

  //@@ make absolutely sure this time has the bunch spacing folded in!
  //@@ and get the geometry service, when it's available
  double jitter = hit.time(); - timeOfFlight(detId);

  const double tzero = parameters.timePhase() -jitter -
     BUNCHSPACE*parameters.binOfMaximum();
  double binTime = tzero;
  CaloSamples result(detId, parameters.readoutFrameSize());
  for(int bin = 0; bin < result.size(); bin++) {
    result[bin] += (*theShape)(binTime)* signal;
   //std::cout << "BIN " << bin << " TIME " << binTime << " SHAPE " << (*theShape)(binTime) << " SIG " << signal << endl;
    binTime += BUNCHSPACE;
  }

  result.setPresamples(parameters.binOfMaximum()-1);
  return result;
} 


double CaloHitResponse::analogSignalAmplitude(const PCaloHit & hit, const CaloSimParameters & parameters) const {
  // OK, the "energy" in the hit could be a real energy, deposited energy,
  // or pe count.  This factor converts to photoelectrons
  int npe = (int)(hit.energy() * parameters.simHitToPhotoelectrons());
  // do we need to doPoisson statistics for the photoelectrons?
  if(parameters.doPhotostatistics()) {
    npe = RandPoisson::shoot(npe);
  }
  // convert to whatever units get read out: charge, voltage, whatever
  return npe * parameters.photoelectronsToAnalog();
}


CaloSamples CaloHitResponse::findSignal(const DetId & detId) const {
  AnalogSignalMap::const_iterator signalItr = theAnalogSignalMap.find(detId);
  if(signalItr == theAnalogSignalMap.end()) {
     // return a blank signal if not found
     const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
     CaloSamples result(detId, parameters.readoutFrameSize());
     result.setPresamples(parameters.binOfMaximum()-1);
     return result;
  } 
  else {
     return signalItr->second;
  }
}


double CaloHitResponse::timeOfFlight(const DetId & detId) const {
  // not going to assume there's one of these per subdetector.
  // Take the whole CaloGeometry and find the right subdet
  double result = 0.;
  if(theGeometry == 0) {
    edm::LogWarning("CaloHitResponse") << "No Calo Geometry set, so no time of flight correction";
  } 
  else {
    const CaloCellGeometry* cellGeometry = theGeometry->getSubdetectorGeometry(detId)->getGeometry(detId);
    double distance = cellGeometry->getPosition().mag();
    result =  distance * cm / c_light; // Units of c_light: mm/ns
  }
  return result;
}


