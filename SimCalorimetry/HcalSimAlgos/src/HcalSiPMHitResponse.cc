#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
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
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "CLHEP/Random/RandPoissonQ.h"

#include <math.h>
#include <list>

HcalSiPMHitResponse::HcalSiPMHitResponse(const CaloVSimParameterMap * parameterMap,
					 const CaloShapes * shapes, bool PreMix1) :
  CaloHitResponse(parameterMap, shapes), theSiPM(), PreMixDigis(PreMix1),
  nbins(BUNCHSPACE*HcalPulseShapes::invDeltaTSiPM_), dt(HcalPulseShapes::deltaTSiPM_), invdt(HcalPulseShapes::invDeltaTSiPM_) {}

HcalSiPMHitResponse::~HcalSiPMHitResponse() {}

void HcalSiPMHitResponse::initializeHits() {
  precisionTimedPhotons.clear();
}

void HcalSiPMHitResponse::finalizeHits(CLHEP::HepRandomEngine* engine) {
  //do not add PE noise for initial premix
  if(!PreMixDigis) addPEnoise(engine);

  photonTimeMap::iterator channelPhotons;
  for (channelPhotons = precisionTimedPhotons.begin();
       channelPhotons != precisionTimedPhotons.end();
       ++channelPhotons) {
    CaloSamples signal(makeSiPMSignal(channelPhotons->first, 
				      channelPhotons->second,
                                      engine));
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

void HcalSiPMHitResponse::add(const PCaloHit& hit, CLHEP::HepRandomEngine* engine) {
    if (!edm::isNotFinite(hit.time()) &&
	((theHitFilter == 0) || (theHitFilter->accepts(hit)))) {
      HcalDetId id(hit.id());
      const HcalSimParameters& pars = dynamic_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));
      //divide out mean of crosstalk distribution 1/(1-lambda) = multiply by (1-lambda)
      double signal(analogSignalAmplitude(id, hit.energy(), pars, engine)*(1-pars.sipmCrossTalk(id)));
      unsigned int photons(signal + 0.5);
      double tof( timeOfFlight(id) );
      double time( hit.time() );
      if(ignoreTime) time = tof;

      if (photons > 0)
        if (precisionTimedPhotons.find(id)==precisionTimedPhotons.end()) {
          precisionTimedPhotons.insert(
            std::pair<DetId, photonTimeHist >(id, 
              photonTimeHist(nbins * pars.readoutFrameSize(), 0)
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
      LogDebug("HcalSiPMHitResponse") << " timePhase: " << pars.timePhase()
		<< " tof: " << tof
		<< " binOfMaximum: " << pars.binOfMaximum()
		<< " phaseShift: " << thePhaseShift_;
      double tzero(0.0 + pars.timePhase() - 
		   (time - tof) - 
		   BUNCHSPACE*( pars.binOfMaximum() - thePhaseShift_));
      LogDebug("HcalSiPMHitResponse") << " tzero: " << tzero;
      double tzero_bin(-tzero*invdt);
      LogDebug("HcalSiPMHitResponse") << " corrected tzero: " << tzero_bin << '\n';
      double t_pe(0.);
      int t_bin(0);
      for (unsigned int pe(0); pe<photons; ++pe) {
        t_pe = HcalPulseShapes::generatePhotonTime(engine);
        t_bin = int(t_pe*invdt + tzero_bin + 0.5);
        LogDebug("HcalSiPMHitResponse") << "t_pe: " << t_pe << " t_pe + tzero: " << (t_pe+tzero_bin*dt)
                  << " t_bin: " << t_bin << '\n';
        if ((t_bin >= 0) && 
            (static_cast<unsigned int>(t_bin) < precisionTimedPhotons[id].size()))
            precisionTimedPhotons[id][t_bin] += 1;
      }
    }
}

void HcalSiPMHitResponse::addPEnoise(CLHEP::HepRandomEngine* engine)
{
  // Add SiPM dark current noise to all cells
  for(std::vector<DetId>::const_iterator idItr = theDetIds->begin();
      idItr != theDetIds->end(); ++idItr) {
    HcalDetId id(*idItr);
    const HcalSimParameters& pars =
      static_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));

    // uA * ns / (fC/pe) = pe!
    double dc_pe_avg =
      pars.sipmDarkCurrentuA(id) * dt / 
      pars.photoelectronsToAnalog(id);

    if (dc_pe_avg <= 0.) continue;

    int nPreciseBins = nbins * pars.readoutFrameSize();

    unsigned int sumnoisePE(0);
    double  elapsedTime(0.);
    for (int tprecise(0); tprecise < nPreciseBins; ++tprecise) {
      int noisepe = CLHEP::RandPoissonQ::shoot(engine, dc_pe_avg); // add dark current noise

      if (noisepe > 0) {
	if (precisionTimedPhotons.find(id)==precisionTimedPhotons.end()) {
	  photonTimeHist photons(nPreciseBins, 0);
	  photons[tprecise] = noisepe;
	  precisionTimedPhotons.insert
	    (std::pair<DetId, photonTimeHist >(id, photons ) );
	} else {
	  precisionTimedPhotons[id][tprecise] += noisepe;
	}

	sumnoisePE += noisepe;
      }
      elapsedTime += dt;

    } // precise time loop

    LogDebug("HcalSiPMHitResponse") << id;
    LogDebug("HcalSiPMHitResponse") << " total noise (PEs): " << sumnoisePE;

  } // detId loop
}                                               // HcalSiPMHitResponse::addPEnoise()

CaloSamples HcalSiPMHitResponse::makeBlankSignal(const DetId& detId) const {
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  int preciseSize(parameters.readoutFrameSize() * nbins);
  CaloSamples result(detId, parameters.readoutFrameSize(), preciseSize);
  result.setPresamples(parameters.binOfMaximum()-1);
  result.setPrecise(result.presamples() * nbins, dt);
  return result;
}

CaloSamples HcalSiPMHitResponse::makeSiPMSignal(DetId const& id, 
						photonTimeHist const& photonTimeBins,
                                                CLHEP::HepRandomEngine* engine) {
  const HcalSimParameters& pars = static_cast<const HcalSimParameters&>(theParameterMap->simParameters(id));  
  theSiPM.setNCells(pars.pixels(id));
  theSiPM.setTau(pars.sipmTau());
  theSiPM.setCrossTalk(pars.sipmCrossTalk(id));
  theSiPM.setSaturationPars(pars.sipmNonlinearity(id));

  //use to make signal
  CaloSamples signal( makeBlankSignal(id) );
  int sampleBin(0), preciseBin(0);
  signal.resetPrecise();
  unsigned int pe(0);
  double hitPixels(0.), elapsedTime(0.);
  unsigned int sumPE(0);
  double sumHits(0.);

  HcalSiPMShape sipmPulseShape(pars.signalShape(id));

  std::list< std::pair<double, double> > pulses;
  std::list< std::pair<double, double> >::iterator pulse;
  double timeDiff, pulseBit;
  LogDebug("HcalSiPMHitResponse") << "makeSiPMSignal for " << HcalDetId(id);

  for (unsigned int tbin(0); tbin < photonTimeBins.size(); ++tbin) {
    pe = photonTimeBins[tbin];
    sumPE += pe;
    preciseBin = tbin;
    sampleBin = preciseBin/nbins;
    if (pe > 0) {
      hitPixels = theSiPM.hitCells(engine, pe, 0., elapsedTime);
      sumHits += hitPixels;
      LogDebug("HcalSiPMHitResponse") << " elapsedTime: " << elapsedTime
				      << " sampleBin: " << sampleBin
				      << " preciseBin: " << preciseBin
				      << " pe: " << pe 
				      << " hitPixels: " << hitPixels ;
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
	LogDebug("HcalSiPMHitResponse") << " pulse t: " << pulse->first 
					<< " pulse A: " << pulse->second
					<< " timeDiff: " << timeDiff
					<< " pulseBit: " << pulseBit;
	signal[sampleBin] += pulseBit;
	signal.preciseAtMod(preciseBin) += pulseBit*invdt;

	if (timeDiff > 1 && sipmPulseShape(timeDiff) < 1e-7)
	  pulse = pulses.erase(pulse);
	else
	  ++pulse;
      }
    }
    elapsedTime += dt;
  }

  return signal;
}

void HcalSiPMHitResponse::setDetIds(const std::vector<DetId> & detIds) {
  theDetIds = &detIds;
}
