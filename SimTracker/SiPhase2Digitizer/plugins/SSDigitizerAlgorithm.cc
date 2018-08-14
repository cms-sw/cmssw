#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/SSDigitizerAlgorithm.h"
#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"


// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

using namespace edm;

void SSDigitizerAlgorithm::init(const edm::EventSetup& es) {
  es.get<TrackerDigiGeometryRecord>().get(geom_);
}
SSDigitizerAlgorithm::SSDigitizerAlgorithm(const edm::ParameterSet& conf) :
  Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
				  conf.getParameter<ParameterSet>("SSDigitizerAlgorithm"))
{
  pixelFlag = false;
  LogInfo("SSDigitizerAlgorithm ") << "SSDigitizerAlgorithm constructed "
				   << "Configuration parameters:"
				   << "Threshold/Gain = "
				   << "threshold in electron Endcap = "
				   << theThresholdInE_Endcap
				   << "threshold in electron Barrel = "
				   << theThresholdInE_Barrel
				   <<" " << theElectronPerADC << " " << theAdcFullScale
				   << " The delta cut-off is set to " << tMax
				   << " pix-inefficiency "<<AddPixelInefficiency;
}
SSDigitizerAlgorithm::~SSDigitizerAlgorithm() {
  LogDebug("SSDigitizerAlgorithm") << "SSDigitizerAlgorithm deleted";
}
void SSDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
					     std::vector<PSimHit>::const_iterator inputEnd,
					     const size_t inputBeginGlobalIndex,
                                             const unsigned int tofBin,
					     const Phase2TrackerGeomDetUnit* pixdet,
					     const GlobalVector& bfield) {
  // produce SignalPoint's for all SimHit's in detector
  // Loop over hits
  uint32_t detId = pixdet->geographicalId().rawId();
  size_t simHitGlobalIndex = inputBeginGlobalIndex; // This needs to be stored to create the digi-sim link later
  for (auto it = inputBegin; it != inputEnd; ++it, ++simHitGlobalIndex) {
    // skip hits not in this detector.
    if ((*it).detUnitId() != detId)
      continue;

    LogDebug("SSDigitizerAlgorithm")
      << (*it).particleType() << " " << (*it).pabs() << " "
      << (*it).energyLoss() << " " << (*it).tof() << " "
      << (*it).trackId() << " " << (*it).processType() << " "
      << (*it).detUnitId()
      << (*it).entryPoint() << " " << (*it).exitPoint() ;
      
    std::vector<DigitizerUtility::EnergyDepositUnit> ionization_points;
    std::vector<DigitizerUtility::SignalPoint> collection_points;

    // fill collection_points for this SimHit, indpendent of topology
    // Check the TOF cut
    if (((*it).tof() - pixdet->surface().toGlobal((*it).localPosition()).mag()/30.) >= theTofLowerCut &&
	((*it).tof() - pixdet->surface().toGlobal((*it).localPosition()).mag()/30.) <= theTofUpperCut) {
      primary_ionization(*it, ionization_points); // fills _ionization_points
      drift(*it, pixdet, bfield, ionization_points, collection_points);  // transforms _ionization_points to collection_points

      // compute induced signal on readout elements and add to _signal

      induce_signal(*it, simHitGlobalIndex, tofBin, pixdet, collection_points); // *ihit needed only for SimHit<-->Digi link
    }
  }
}
