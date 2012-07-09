#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <set>
#include <map>

//FIXME

class PCaloHitCompareTimes {
public:
  bool operator()(const PCaloHit * a, 
		  const PCaloHit * b) const {
    return a->time()<b->time();
  }
};

HcalSiPMHitResponse::HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap,
					 const CaloVShape * shape, const CaloVShape * integratedShape) :
  CaloHitResponse(parameterMap, shape), theIntegratedShape(integratedShape), theSiPM(0), theRecoveryTime(250.) {
  theSiPM = new HcalSiPM(14000);
  // normalize the shape to a peak of 1, not an integral of 1
  theShapeNormalization = 1./(*shape)(shape->timeToRise());
}

HcalSiPMHitResponse::~HcalSiPMHitResponse() {
  if (theSiPM)
    delete theSiPM;
}

void HcalSiPMHitResponse::run(MixCollection<PCaloHit> & hits) {

  typedef std::multiset <const PCaloHit *, PCaloHitCompareTimes> SortedHitSet;

  std::map< DetId, SortedHitSet > sortedhits;
  for (MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
       hitItr != hits.end(); ++hitItr) {
    if (!((hitItr.bunch() < theMinBunch) || (hitItr.bunch() > theMaxBunch)) &&
	!(isnan(hitItr->time())) &&
	((theHitFilter == 0) || (theHitFilter->accepts(*hitItr)))) {
      DetId id(hitItr->id());
      if (sortedhits.find(id)==sortedhits.end())
	sortedhits.insert(std::pair<DetId, SortedHitSet>(id, SortedHitSet()));
      sortedhits[id].insert(&(*hitItr));
    }
  }
  int pixelIntegral, oldIntegral;
  HcalSiPMRecovery pixelHistory(theRecoveryTime);
  const PCaloHit * hit;
  for (std::map<DetId, SortedHitSet>::iterator i = sortedhits.begin(); 
       i!=sortedhits.end(); ++i) {
    pixelHistory.clearHistory();
    for (SortedHitSet::iterator itr = i->second.begin(); 
	 itr != i->second.end(); ++itr) {
      hit = *itr;
      pixelIntegral = pixelHistory.getIntegral(hit->time());
      oldIntegral = pixelIntegral;
      add( makeSiPMSignal(*hit, pixelIntegral) );
      pixelHistory.addToHistory(hit->time(), pixelIntegral-oldIntegral);
    }
  }
}


void HcalSiPMHitResponse::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theSiPM->initRandomEngine(engine);
  CaloHitResponse::setRandomEngine(engine);
}


CaloSamples HcalSiPMHitResponse::makeBlankSignal(const DetId & detId) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  int nSamples = parameters.readoutFrameSize();
  int preciseSize = nSamples * theTDCParameters.nbins();
  CaloSamples result(detId, nSamples, preciseSize);
  result.setPresamples(parameters.binOfMaximum()-1);
  result.setPrecise(0, theTDCParameters.deltaT());
  return result;
}


CaloSamples HcalSiPMHitResponse::makeSiPMSignal(const PCaloHit & inHit, 
						int & integral ) const {
  PCaloHit hit = inHit;
  if (theHitCorrection != 0)
    theHitCorrection->correct(hit);

  HcalDetId id(hit.id());
  const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
  theSiPM->setNCells(pars.pixels());

  double signal = analogSignalAmplitude(hit, pars);
  int photons = static_cast<int>(signal + 0.5);
  int pixels = theSiPM->hitCells(photons, integral);
  integral += pixels;
  signal = double(pixels);

  CaloSamples result(makeBlankSignal(id));
  if(pixels > 0)
  {
    double jitter = hit.time() - timeOfFlight(id);

    const double tzero = pars.timePhase() - jitter -
      BUNCHSPACE*(pars.binOfMaximum() - thePhaseShift_);

    double binTime = tzero;
    int nbins =  result.size();
    for (int bin = 0; bin < nbins; ++bin) {
      result[bin] += (*theIntegratedShape)(binTime)*signal;
      // fill precise
      double preciseTime = binTime;
      int preciseBin = bin * theTDCParameters.nbins();
      for(int preciseBXBin = 0; preciseBXBin < theTDCParameters.nbins(); ++preciseBXBin)
      {
        result.preciseAtMod(preciseBin) = (*theShape)(preciseTime) * signal * theShapeNormalization;
        ++preciseBin;
        preciseTime += theTDCParameters.deltaT();
      }        
      binTime += BUNCHSPACE;
    }
  }

  return result;
}


