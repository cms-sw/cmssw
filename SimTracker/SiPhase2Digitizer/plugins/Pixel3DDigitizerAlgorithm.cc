#include "SimTracker/SiPhase2Digitizer/plugins/Pixel3DDigitizerAlgorithm.h"

//#include "SimTracker/Common/interface/SiG4UniversalFluctuation.h"
//#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"

//#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
//#include "CLHEP/Random/RandGaussQ.h"
//#include "CLHEP/Random/RandFlat.h"

// Framework infrastructure
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/Utilities/interface/Exception.h"

// Calibration & Conditions
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"
//#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
//#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
//#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
//#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
//#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
//#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
//#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
//#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#include <iostream>
//#include <cmath>

using namespace sipixelobjects;

// REMEMBER CMS conventions:
// -- Energy: GeV
// -- momentum: GeV/c
// -- mass: GeV/c^2
// -- Distance, position: cm
// -- Time: ns
// -- Angles: radian
// Some constants 
const double SPEED_OF_LIGHT=30.0;

void Pixel3DDigitizerAlgorithm::init(const edm::EventSetup& es) 
{
    // XXX: Just copied from PixelDigitizer Algorithm
    //      CHECK if all these is needed


    if(use_ineff_from_db_) 
    {
        // load gain calibration service fromdb...
        theSiPixelGainCalibrationService_->setESObjects(es);
    }

    if(use_deadmodule_DB_)
    {
        es.get<SiPixelQualityRcd>().get(SiPixelBadModule_);
    }

    if(use_LorentzAngle_DB_)
    {  
        // Get Lorentz angle from DB record
        es.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);
    }

    // gets the map and geometry from the DB (to kill ROCs)
    es.get<SiPixelFedCablingMapRcd>().get(map_);
    es.get<TrackerDigiGeometryRecord>().get(geom_);
}

Pixel3DDigitizerAlgorithm::Pixel3DDigitizerAlgorithm(const edm::ParameterSet& conf) : 
    Phase2TrackerDigitizerAlgorithm(
            conf.getParameter<edm::ParameterSet>("AlgorithmCommon"),
            conf.getParameter<edm::ParameterSet>("Pixel3DDigitizerAlgorithm")) 
{
    // XXX - NEEDED?
    pixelFlag = true;

    edm::LogInfo("Pixel3DDigitizerAlgorithm") << "Algorithm constructed "
                                     << "Configuration parameters:"
                                     << "Threshold/Gain = "
                                     << "threshold in electron Endcap = " << theThresholdInE_Endcap
                                     << "threshold in electron Barrel = " << theThresholdInE_Barrel << " "
                                     << theElectronPerADC << " " << theAdcFullScale << " The delta cut-off is set to "
                                     << tMax << " pix-inefficiency " << AddPixelInefficiency;
}


Pixel3DDigitizerAlgorithm::~Pixel3DDigitizerAlgorithm() 
{ 
    LogDebug("Pixel3DDigitizerAlgorithm") << "Algorithm deleted"; 
}

void Pixel3DDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                                std::vector<PSimHit>::const_iterator inputEnd,
                                                const size_t inputBeginGlobalIndex,
                                                const unsigned int tofBin,
                                                const Phase2TrackerGeomDetUnit* pix3Ddet,
                                                const GlobalVector& bfield) 
{
    // produce SignalPoint's for all SimHit's in detector
    
    const uint32_t detId = pix3Ddet->geographicalId().rawId();
    // This needs to be stored to create the digi-sim link later
    size_t simHitGlobalIndex = inputBeginGlobalIndex;  
    
    // Loop over hits
    for(auto it = inputBegin; it != inputEnd; ++it, ++simHitGlobalIndex) 
    {
        // skip hit: not in this detector.
        if(it->detUnitId() != detId)
        {
            continue;
        }

        LogDebug("Pixel3DDigitizerAlgorithm") 
            << (*it).particleType() << " " << (*it).pabs() << " " << (*it).energyLoss()
            << " " << (*it).tof() << " " << (*it).trackId() << " " << (*it).processType()
            << " " << (*it).detUnitId() << (*it).entryPoint() << " " << (*it).exitPoint();

        // Convert the simhit position into global to check if the simhit was 
        // produced within a given time-window
        const auto global_hit_position = pix3Ddet->surface().toGlobal(it->localPosition()).mag();

        // Only accept those sim hits produced inside a time window (same bunch-crossing)
        if( (it->tof()-global_hit_position/SPEED_OF_LIGHT >= theTofLowerCut) 
                && (it->tof()-global_hit_position/SPEED_OF_LIGHT <= theTofUpperCut) )
        {
            // XXX: this vectors are the output of the next methods, the methods should
            // return them, instead of an input argument
            std::vector<DigitizerUtility::EnergyDepositUnit> ionization_points;
            std::vector<DigitizerUtility::SignalPoint> collection_points;

            // For each sim hit, super-charges (electron-holes) are created every 10um
            primary_ionization(*it, ionization_points);
            // Drift the super-charges (only electrons) to the collecting electrodes
            drift(*it, pix3Ddet, bfield, ionization_points, collection_points);
            
            // compute induced signal on readout elements and add to _signal
            // *ihit needed only for SimHit<-->Digi link
            induce_signal(*it, simHitGlobalIndex, tofBin, pix3Ddet, collection_points);
        }
    }
}
