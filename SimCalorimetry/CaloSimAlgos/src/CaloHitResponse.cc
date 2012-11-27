#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h" 
#include "SimDataFormats/CaloHit/interface/PCaloHit.h" 
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 

#include<iostream>

CaloHitResponse::CaloHitResponse(const CaloVSimParameterMap * parametersMap, 
                                 const CaloVShape * shape)
: theAnalogSignalMap(),
  theParameterMap(parametersMap), 
  theShapes(0),  
  theShape(shape),  
  theHitCorrection(0),
  thePECorrection(0),
  theHitFilter(0),
  theGeometry(0),
  theRandPoisson(0),
  theMinBunch(-10), 
  theMaxBunch(10),
  thePhaseShift_(1.),
  changeScale(false) {}

CaloHitResponse::CaloHitResponse(const CaloVSimParameterMap * parametersMap,
                                 const CaloShapes * shapes)
: theAnalogSignalMap(),
  theParameterMap(parametersMap),
  theShapes(shapes),
  theShape(0),
  theHitCorrection(0),
  thePECorrection(0),
  theHitFilter(0),
  theGeometry(0),
  theRandPoisson(0),
  theMinBunch(-10),
  theMaxBunch(10),
  thePhaseShift_(1.),
  changeScale(false) {}

CaloHitResponse::~CaloHitResponse() {
  delete theRandPoisson;
}

void CaloHitResponse::initHBHEScale() {
#ifdef ChangeHcalEnergyScale
  for (int ij=0; ij<100; ij++) {
    for (int jk=0; jk<72; jk++) {	
      for (int kl=0; kl<4; kl++) {
	hcal_en_scale[ij][jk][kl] = 1.0;
      }
    }
  }
#endif
}

void CaloHitResponse::setHBHEScale(std::string & fileIn) {
  
  ifstream infile(fileIn.c_str());
  LogDebug("CaloHitResponse") << "Reading from " << fileIn;
#ifdef ChangeHcalEnergyScale
  if (!infile.is_open()) {
    edm::LogError("CaloHitResponse") << "** ERROR: Can't open '" << fileIn << "' for the input file";
  } else {
    int     eta, phi, depth;
    double  cFactor;
    while(1) {
      infile >> eta >> phi >> depth >> cFactor;
      if (!infile.good()) break;
      hcal_en_scale[eta][phi][depth] = cFactor;
      //      LogDebug("CaloHitResponse") << "hcal_en_scale[" << eta << "][" << phi << "][" << depth << "] = " << hcal_en_scale[eta][phi][depth];
    }
    infile.close();
  }
  changeScale = true;
#endif
}

void CaloHitResponse::setBunchRange(int minBunch, int maxBunch) {
  theMinBunch = minBunch;
  theMaxBunch = maxBunch;
}


void CaloHitResponse::setRandomEngine(CLHEP::HepRandomEngine & engine) {
  theRandPoisson = new CLHEP::RandPoissonQ(engine);
}


void CaloHitResponse::run(MixCollection<PCaloHit> & hits) {

  for(MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr) {
    if(withinBunchRange(hitItr.bunch())) {
      add(*hitItr);
    } // loop over hits
  }
}

void CaloHitResponse::add( const PCaloHit& hit ) {
  // check the hit time makes sense
  if ( edm::isNotFinite(hit.time()) ) { return; }

  // maybe it's not from this subdetector
  if(theHitFilter == 0 || theHitFilter->accepts(hit)) {
    LogDebug("CaloHitResponse") << hit;
    CaloSamples signal( makeAnalogSignal( hit ) ) ;

    bool keep ( keepBlank() ) ;  // here we  check for blank signal if not keeping them
    if( !keep )
    {
       const unsigned int size ( signal.size() ) ;
       if( 0 != size )
       {
	  for( unsigned int i ( 0 ) ; i != size ; ++i )
	  {
	     keep = keep || signal[i] > 1.e-7 ;
	  }
       }
    }
    LogDebug("CaloHitResponse") << signal;
    if( keep ) add(signal);
  }
}


void CaloHitResponse::add(const CaloSamples & signal)
{
  DetId id(signal.id());
  CaloSamples * oldSignal = findSignal(id);
  if (oldSignal == 0) {
    theAnalogSignalMap[id] = signal;
  } else  {
    // need a "+=" to CaloSamples
    int sampleSize =  oldSignal->size();
    assert(sampleSize == signal.size());
    assert(signal.presamples() == oldSignal->presamples());

    for(int i = 0; i < sampleSize; ++i) {
      (*oldSignal)[i] += signal[i];
    }
  }
}


CaloSamples CaloHitResponse::makeAnalogSignal(const PCaloHit & hit) const {

  DetId detId(hit.id());
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  
  double signal = analogSignalAmplitude(detId, hit.energy(), parameters);

  double time = hit.time();
  if(theHitCorrection != 0) {
    time += theHitCorrection->delay(hit);
  }
  double jitter = hit.time() - timeOfFlight(detId);

  const CaloVShape * shape = theShape;
  if(!shape) {
    shape = theShapes->shape(detId);
  }
  // assume bins count from zero, go for center of bin
  const double tzero = ( shape->timeToRise()
			 + parameters.timePhase() 
			 - jitter 
			 - BUNCHSPACE*( parameters.binOfMaximum()
					- thePhaseShift_          ) ) ;
  double binTime = tzero;

  CaloSamples result(makeBlankSignal(detId));

  for(int bin = 0; bin < result.size(); bin++) {
    result[bin] += (*shape)(binTime)* signal;
    binTime += BUNCHSPACE;
  }
  return result;
} 


double CaloHitResponse::analogSignalAmplitude(const DetId & detId, float energy, const CaloSimParameters & parameters) const {

  if(!theRandPoisson)
    {
      edm::Service<edm::RandomNumberGenerator> rng;
      if ( ! rng.isAvailable()) {
	throw cms::Exception("Configuration")
	  << "CaloHitResponse requires the RandomNumberGeneratorService\n"
	  "which is not present in the configuration file.  You must add the service\n"
	  "in the configuration file or remove the modules that require it.";
      }
      theRandPoisson = new CLHEP::RandPoissonQ(rng->getEngine());
    }
  
  // OK, the "energy" in the hit could be a real energy, deposited energy,
  // or pe count.  This factor converts to photoelectrons
  //GMA Smeared in photon production it self  
  double scl =1.0;
#ifdef ChangeHcalEnergyScale
  if (changeScale) {
    if (detId.det()==DetId::Hcal ) { 
      HcalDetId dId = HcalDetId(detId); 
      if (dId.subdet()==HcalBarrel || dId.subdet()==HcalEndcap) { 
	int ieta = dId.ieta()+50;
	int iphi = dId.iphi()-1;
	int idep = dId.depth()-1;
	scl = hcal_en_scale[ieta][iphi][idep];
	LogDebug("CaloHitResponse") << " ID " << dId << " Scale " << scl;
      }
    }
  } 
#endif
  double npe = scl * energy * parameters.simHitToPhotoelectrons(detId);
  // do we need to doPoisson statistics for the photoelectrons?
  if(parameters.doPhotostatistics()) {
    npe = theRandPoisson->fire(npe);
  }
  if(thePECorrection) npe = thePECorrection->correctPE(detId, npe);
  return npe;
}


CaloSamples * CaloHitResponse::findSignal(const DetId & detId) {
  CaloSamples * result = 0;
  AnalogSignalMap::iterator signalItr = theAnalogSignalMap.find(detId);
  if(signalItr == theAnalogSignalMap.end()) {
    result = 0;
  } else {
    result = &(signalItr->second);
  }
  return result;
}


CaloSamples CaloHitResponse::makeBlankSignal(const DetId & detId) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  CaloSamples result(detId, parameters.readoutFrameSize());
  result.setPresamples(parameters.binOfMaximum()-1);
  return result;
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
    if(cellGeometry == 0) {
       edm::LogWarning("CaloHitResponse") << "No Calo cell found for ID"
         << detId.rawId() << " so no time-of-flight subtraction will be done";
    }
    else {
      double distance = cellGeometry->getPosition().mag();
      result =  distance * cm / c_light; // Units of c_light: mm/ns
    }
  }
  return result;
}


