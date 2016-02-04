#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

#include <set>
#include <map>

class PCaloHitCompareTimes {
public:
  bool operator()(const PCaloHit * a, 
		  const PCaloHit * b) const {
    return a->time()<b->time();
  }
};

HcalSiPMHitResponse::HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap,
					 const CaloShapes * shapes) :
  CaloHitResponse(parameterMap, shapes), theSiPM(0), theRecoveryTime(250.) {
  theSiPM = new HcalSiPM(14000);
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
      CaloSamples signal(makeSiPMSignal(*hit, pixelIntegral));
      pixelHistory.addToHistory(hit->time(), pixelIntegral-oldIntegral);
      add(signal);
    }
  }
}


void HcalSiPMHitResponse::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theSiPM->initRandomEngine(engine);
  CaloHitResponse::setRandomEngine(engine);
}


CaloSamples HcalSiPMHitResponse::makeSiPMSignal(const PCaloHit & inHit, 
						int & integral ) const {
  PCaloHit hit = inHit;
  if (theHitCorrection != 0)
    theHitCorrection->correct(hit);

  DetId id(hit.id());
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
    const CaloVShape * shape = theShapes->shape(id);
    double jitter = hit.time() - timeOfFlight(id);

    const double tzero = pars.timePhase() - jitter -
      BUNCHSPACE*(pars.binOfMaximum() - thePhaseShift_);
    double binTime = tzero;

    for (int bin = 0; bin < result.size(); bin++) {
      result[bin] += (*shape)(binTime)*signal;
      binTime += BUNCHSPACE;
    }
  }

  return result;
}

