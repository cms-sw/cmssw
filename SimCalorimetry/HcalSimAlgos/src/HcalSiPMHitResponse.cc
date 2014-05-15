#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPM.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameters.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"

#include <math.h>
#include <list>

HcalSiPMHitResponse::HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap,
					 const CaloShapes * shapes) :
  CaloHitResponse(parameterMap, shapes), theSiPM(), theRecoveryTime(250.), 
  TIMEMULT(1), Y11RANGE(80.), Y11MAX(0.04), Y11TIMETORISE(16.65), 
  theRndFlat(0) {
  theSiPM = new HcalSiPM(2500);
}

HcalSiPMHitResponse::~HcalSiPMHitResponse() {
  if (theSiPM)
    delete theSiPM;
  delete theRndFlat;
}

void HcalSiPMHitResponse::initializeHits() {
  precisionTimedPhotons.clear();
}

void HcalSiPMHitResponse::finalizeHits() {

  photonTimeMap::iterator channelPhotons;
  for (channelPhotons = precisionTimedPhotons.begin();
       channelPhotons != precisionTimedPhotons.end();
       ++channelPhotons) {
    CaloSamples signal(makeSiPMSignal(channelPhotons->first, 
				      channelPhotons->second));
    bool keep( keepBlank() );
    if (!keep) {
      const unsigned int size ( signal.size() ) ;
      if( 0 != size ) {
	for( unsigned int i ( 0 ) ; i != size ; ++i ) {
	  keep = keep || signal[i] > 1.e-7 ;
	}
      }
    }

    LogDebug("HcalSiPMHitResponse") << HcalDetId(signal.id()) << ' ' << signal;
    // std::cout << HcalDetId(signal.id()) << ' ' << signal;
    if (keep) add(signal);
  }
}

void HcalSiPMHitResponse::add(const CaloSamples& signal) {
  DetId id(signal.id());
  CaloSamples * oldSignal = findSignal(id);
  if (oldSignal == 0) {
    theAnalogSignalMap[id] = signal;
  } else {
    (*oldSignal) += signal;
  }
}

void HcalSiPMHitResponse::add(const PCaloHit& hit) {
    if (!edm::isNotFinite(hit.time()) &&
	((theHitFilter == 0) || (theHitFilter->accepts(hit)))) {
      HcalDetId id(hit.id());
      const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
      double signal(analogSignalAmplitude(id, hit.energy(), pars));
      unsigned int photons(signal + 0.5);
      double time( hit.time() );

      if (photons > 0)
	if (precisionTimedPhotons.find(id)==precisionTimedPhotons.end()) {
	  precisionTimedPhotons.insert(
	    std::pair<DetId, photonTimeHist >(id, 
	      photonTimeHist(theTDCParams.nbins() * TIMEMULT *
			     pars.readoutFrameSize(), 0)
					      )
				       );
	}

      LogDebug("HcalSiPMHitResponse") << id;
      LogDebug("HcalSiPMHitResponse") << " fCtoGeV: " << pars.fCtoGeV(id)
		<< " samplingFactor: " << pars.samplingFactor(id)
		<< " photoelectronsToAnalog: " << pars.photoelectronsToAnalog(id)
		<< " simHitToPhotoelectrons: " << pars.simHitToPhotoelectrons(id);
      LogDebug("HcalSiPMHitResponse") << " energy: " << hit.energy()
		<< " photons: " << photons 
		<< " time: " << time;
      if (theHitCorrection != 0)
	time += theHitCorrection->delay(hit);
      LogDebug("HcalSiPMHitResponse") << " corrected time: " << time;
      LogDebug("HcalSiPMHitResponse") << " timePhase: " << pars.timePhase()
		<< " tof: " << timeOfFlight(id)
		<< " binOfMaximum: " << pars.binOfMaximum()
		<< " phaseShift: " << thePhaseShift_;
      double tzero(Y11TIMETORISE + pars.timePhase() - 
		   (hit.time() - timeOfFlight(id)) - 
		   BUNCHSPACE*( pars.binOfMaximum() - thePhaseShift_));
      LogDebug("HcalSiPMHitResponse") << " tzero: " << tzero;
      // tzero += BUNCHSPACE*pars.binOfMaximum() + 75.;
      //move it back 25ns to bin 4
      tzero += BUNCHSPACE*pars.binOfMaximum() + 50.;
      LogDebug("HcalSiPMHitResponse") << " corrected tzero: " << tzero << '\n';
      double t_pe(0.);
      int t_bin(0);
      for (unsigned int pe(0); pe<photons; ++pe) {
	t_pe = generatePhotonTime();
	t_bin = int((t_pe + tzero)/(theTDCParams.deltaT()/TIMEMULT) + 0.5);
	LogDebug("HcalSiPMHitResponse") << "t_pe: " << t_pe << " t_pe + tzero: " << (t_pe+tzero)
		  << " t_bin: " << t_bin << '\n';
	if ((t_bin >= 0) && 
	    (static_cast<unsigned int>(t_bin) < precisionTimedPhotons[id].size()))
	    precisionTimedPhotons[id][t_bin] += 1;
      }
    }
}

void HcalSiPMHitResponse::run(MixCollection<PCaloHit> & hits) {
  typedef std::multiset <const PCaloHit *, PCaloHitCompareTimes> SortedHitSet;

  std::map< DetId, SortedHitSet > sortedhits;
  for (MixCollection<PCaloHit>::MixItr hitItr = hits.begin();
       hitItr != hits.end(); ++hitItr) {
    if (!((hitItr.bunch() < theMinBunch) || (hitItr.bunch() > theMaxBunch)) &&
        !(edm::isNotFinite(hitItr->time())) &&
        ((theHitFilter == 0) || (theHitFilter->accepts(*hitItr)))) {
      DetId id(hitItr->id());
      if (sortedhits.find(id)==sortedhits.end())
        sortedhits.insert(std::pair<DetId, SortedHitSet>(id, SortedHitSet()));
      sortedhits[id].insert(&(*hitItr));
    }
  }
  int pixelIntegral, oldIntegral;
  HcalSiPMRecovery pixelHistory(theRecoveryTime);
  for (std::map<DetId, SortedHitSet>::iterator i = sortedhits.begin(); 
       i!=sortedhits.end(); ++i) {
    pixelHistory.clearHistory();
    for (SortedHitSet::iterator itr = i->second.begin(); 
	 itr != i->second.end(); ++itr) {
      const PCaloHit& hit = **itr;
      pixelIntegral = pixelHistory.getIntegral(hit.time());
      oldIntegral = pixelIntegral;
      CaloSamples signal(makeSiPMSignal(i->first, hit, pixelIntegral));
      pixelHistory.addToHistory(hit.time(), pixelIntegral-oldIntegral);
      add(signal);
    }
  }
}


void HcalSiPMHitResponse::setRandomEngine(CLHEP::HepRandomEngine & engine)
{
  theSiPM->initRandomEngine(engine);
  CaloHitResponse::setRandomEngine(engine);
  theRndFlat = new CLHEP::RandFlat(engine);
}

CaloSamples HcalSiPMHitResponse::makeBlankSignal(const DetId& detId) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  int preciseSize(parameters.readoutFrameSize()*theTDCParams.nbins());
  CaloSamples result(detId, parameters.readoutFrameSize(), preciseSize);
  result.setPresamples(parameters.binOfMaximum()-1);
  result.setPrecise(result.presamples()*theTDCParams.nbins(), 
		    theTDCParams.deltaT());
  return result;
}

CaloSamples HcalSiPMHitResponse::makeSiPMSignal(const DetId & id,
                                                const PCaloHit & inHit, 
						int & integral ) const {
  
  PCaloHit hit = inHit;
  if (theHitCorrection != 0) {
    hit.setTime(hit.time() + theHitCorrection->delay(hit));
  }

  const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
  theSiPM->setNCells(pars.pixels());

  double signal = analogSignalAmplitude(id, hit.energy(), pars);
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

CaloSamples HcalSiPMHitResponse::makeSiPMSignal(DetId const& id, 
						photonTimeHist const& photons) const {
  const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));  
  theSiPM->setNCells(pars.pixels());
  theSiPM->setTau(5.);
  //use to make signal
  CaloSamples signal( makeBlankSignal(id) );
  double const dt(theTDCParams.deltaT()/TIMEMULT);
  double const invdt(1./theTDCParams.deltaT());
  int sampleBin(0), preciseBin(0);
  signal.resetPrecise();
  unsigned int pe(0);
  double hitPixels(0.), elapsedTime(0.);
  unsigned int sumPE(0);
  double sumHits(0.);

  HcalSiPMShape sipmPulseShape;

  std::list< std::pair<double, double> > pulses;
  std::list< std::pair<double, double> >::iterator pulse;
  double timeDiff, pulseBit;
  // std::cout << HcalDetId(id) << '\n';
  for (unsigned int pt(0); pt < photons.size(); ++pt) {
    pe = photons[pt];
    sumPE += pe;
    preciseBin = pt/TIMEMULT;
    sampleBin = preciseBin/theTDCParams.nbins();
    if (pe > 0) {
      hitPixels = theSiPM->hitCells(pe, 0., elapsedTime);
      sumHits += hitPixels;
      // std::cout << " elapsedTime: " << elapsedTime
      // 		<< " sampleBin: " << sampleBin
      // 		<< " preciseBin: " << preciseBin
      // 		<< " pe: " << pe 
      // 		<< " hitPixels: " << hitPixels 
      // 		<< '\n';
      if (pars.doSiPMSmearing()) {
	pulses.push_back( std::pair<double, double>(elapsedTime, hitPixels) );
      } else {
	signal[sampleBin] += hitPixels;
	hitPixels *= invdt;
	signal.preciseAtMod(preciseBin) += 0.6*hitPixels;
	if (preciseBin > 0)
	  signal.preciseAtMod(preciseBin-1) += 0.2*hitPixels;
	if (preciseBin < signal.preciseSize() -1)
	  signal.preciseAtMod(preciseBin+1) += 0.2*hitPixels;
      }
    }
    
    if (pars.doSiPMSmearing()) {
      pulse = pulses.begin();
      while (pulse != pulses.end()) {
	timeDiff = elapsedTime - pulse->first;
	pulseBit = sipmPulseShape(timeDiff)*pulse->second;
	// std::cout << "pulse t: " << pulse->first 
	// 		<< " pulse A: " << pulse->second
	// 		<< " timeDiff: " << timeDiff
	// 		<< " pulseBit: " << pulseBit 
	// 		<< '\n';
	signal[sampleBin] += pulseBit;
	signal.preciseAtMod(preciseBin) += pulseBit*invdt;
	if (sipmPulseShape(timeDiff) < 1e-6)
	  pulse = pulses.erase(pulse);
	else
	  ++pulse;
      }
    }
    elapsedTime += dt;
  }

  // differentiatePreciseSamples(signal, 1.);

  // std::cout << "sum pe: " << sumPE 
  // 	    << " sum sipm pixels: " << sumHits
  // 	    << std::endl;
  
  return signal;
}

void HcalSiPMHitResponse::differentiatePreciseSamples(CaloSamples& samples,
						      double diffNorm) const {
  static double const invdt(1./samples.preciseDeltaT());
  // double dy(0.);
  for (int i(0); i < samples.preciseSize(); ++i) {
    // dy = samples.preciseAt(i+1) - samples.preciseAt(i);
    samples.preciseAtMod(i) *= invdt*diffNorm;
  }
}

double HcalSiPMHitResponse::generatePhotonTime() const {
  double result(0.);
  while (true) {
    result = theRndFlat->fire(Y11RANGE);
    if (theRndFlat->fire(Y11MAX) < Y11TimePDF(result))
      return result;
  }
}

double HcalSiPMHitResponse::Y11TimePDF(double t) {
  return exp(-0.0635-0.1518*t)*pow(t, 2.528)/2485.9;
}
