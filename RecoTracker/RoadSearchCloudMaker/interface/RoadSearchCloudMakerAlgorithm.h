#ifndef CloudMakerAlgorithm_h
#define CloudMakerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchCloudMaker
// Class:           RoadSearchCloudMakerAlgorithm
// 
// Description:     
//                  Road categories determined by outer Seed RecHit
//                  	RPhi: outer Seed RecHit in the Barrel
//                  	ZPhi: outer Seed RecHit in the Disks
//                  use inner and outer Seed RecHit and BeamSpot to calculate extrapolation
//                  	RPhi: phi = phi0 + asin(k r)
//                  	ZPhi: phi = phi0 + C z
//                  Loop over RoadSet, access Rings of Road
//                  	get average radius of Ring
//                  	use extrapolation to calculate phi_ref at average Ring radius
//                  	determine window in phi for DetId lookup in the Ring
//                  		phi_ref ± phi_window
//                  			PARAMETER: phi_window = pi/24
//                  	loop over DetId's in phi window in Ring
//                  		two cases (problem of phi = 0/2pi):
//                  			lower window border < upper window border
//                  			upper window border < lower window border
//                  Loop over RecHits of DetId
//                  	check compatibility of RecHit with extrapolation (delta phi cut)
//                  	single layer sensor treatment
//                  		RPhi:
//                  			stereo and barrel single layer sensor
//                  				calculate delta-phi
//                  			disk single layer sensor (r coordinate not well defined)
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  		ZPhi:
//                  			stereo sensor
//                  				calculate delta-phi
//                  			barrel single layer sensor (z coordinate not well defined)
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  			disk single layer sensor (tilted strips relative to local coordinate system of sensor
//                  				calculate delta phi between global positions of maximal strip extension and reference
//                  Check delta phi cut
//                  	cut value can be calculated based on extrapolation and Seed (pT dependent delta phi cut)
//                  	currently: constant delta phi cut (PARAMETER)
//                  		fill RecHit into Cloud
//                  			do not fill more than 32 RecHits per DetID into cloud (PARAMETER)
//                  first stage of Cloud cleaning cuts:
//                  	number of layers with hits in cloud (PARAMETER)
//                  	number of layers with no hits in cloud (PARAMETER)
//                  	number of consecutive layers with no hits in cloud (PARAMETER)
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2006/01/14 22:00:00 $
// $Revision: 1.1 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackingSeed/interface/TrackingSeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"

#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

class RoadSearchCloudMakerAlgorithm 
{
 public:
  
  RoadSearchCloudMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchCloudMakerAlgorithm();

  /// Runs the algorithm
  void run(const TrackingSeedCollection* input,
	   const SiStripRecHit2DLocalPosCollection* inputRecHits,
	   const edm::EventSetup& es,
	   RoadSearchCloudCollection &output);

  void FillRecHitsIntoCloud(DetId id, const SiStripRecHit2DLocalPosCollection* inputRecHits, 
			    double phi0, double k0, Roads::type roadType, double ringPhi,
			    const TrackingSeed* seed, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector,
			    const TrackingGeometry *tracker, RoadSearchCloud &cloud);

  bool isSingleLayer(DetId id);

  bool isBarrelSensor(DetId id);

  double phiFromExtrapolation(double phi0, double k0, double ringRadius, Roads::type roadType);

  double phiMax(const TrackingSeed *seed, double phi0, double k0);

  double map_phi(double phi);

  void setLayerNumberArray(DetId id, std::vector<bool> &usedLayersArray, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector);

  unsigned int getIndexInUsedLayersArray(DetId id, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector);

  bool checkMinimalNumberOfUsedLayers(std::vector<bool> &usedLayersArray);
  bool checkMaximalNumberOfMissedLayers(std::vector<bool> &usedLayersArray, const Roads::RoadSet &roadSet, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector);
  bool checkMaximalNumberOfConsecutiveMissedLayers(std::vector<bool> &usedLayersArray, const Roads::RoadSet &roadSet, Roads::NumberOfLayersPerSubdetector &numberOfLayersPerSubdetector);

 private:
  edm::ParameterSet conf_;

};

#endif
