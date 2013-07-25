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
// $Author: burkett $
// $Date: 2007/08/30 14:59:11 $
// $Revision: 1.28 $
//

#include <string>
#include <sstream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/RoadSearchSeed/interface/RoadSearchSeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "RecoTracker/RoadMapRecord/interface/Roads.h"

#include "TrackingTools/RoadSearchHitAccess/interface/DetHitAccess.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
 
class RoadSearchCloudMakerAlgorithm 
{
 public:
  
  RoadSearchCloudMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchCloudMakerAlgorithm();

  /// Runs the algorithm
  void run(edm::Handle<RoadSearchSeedCollection> input,
	   const SiStripRecHit2DCollection* rphiRecHits,
	   const SiStripRecHit2DCollection* stereoRecHits,
	   const SiStripMatchedRecHit2DCollection* matchedRecHits,
	   const SiPixelRecHitCollection *pixRecHits,
	   const edm::EventSetup& es,
	   RoadSearchCloudCollection &output);

  unsigned int FillRecHitsIntoCloudGeneral(DetId id, double d0, double phi0, double k0, double phi1, double k1,
					   Roads::type roadType, double ringPhi,
					   const TrackerGeometry *tracker, const SiStripRecHitMatcher* theHitMatcher, RoadSearchCloud &cloud);

  unsigned int FillRecHitsIntoCloud(DetId id, const SiStripRecHit2DCollection* inputRecHits, 
				    double d0, double phi0, double k0, Roads::type roadType, double ringPhi,
				    const TrackerGeometry *tracker, RoadSearchCloud &cloud);

  
  unsigned int FillPixRecHitsIntoCloud(DetId id, 
				       const SiPixelRecHitCollection *inputRecHits, 
				       double d0, double phi0, double k0, Roads::type roadType, double ringPhi,
				       const TrackerGeometry *tracker, RoadSearchCloud &cloud);

  bool isSingleLayer(DetId id);

  bool isBarrelSensor(DetId id);

  double phiFromExtrapolation(double d0, double phi0, double k0, double ringRadius, Roads::type roadType);

  double phiMax(Roads::type roadType, double phi0, double k0);

  double map_phi(double phi);
  double map_phi2(double phi);

  void makecircle(double x1_cs, double y1_cs,double x2_cs, double y2_cs,
                                             double x3_cs, double y3_cs);

  double CheckXYIntersection(LocalPoint& ip1, LocalPoint& op1, LocalPoint& ip2, LocalPoint& op2);

  double CheckZPhiIntersection(double iPhi1, double iZ1, double oPhi1, double oZ1,
			       double iPhi2, double iZ2, double oPhi2, double oZ2);

  double ZPhiDeltaPhi(double phi1, double phi2, double phiExpect);

  RoadSearchCloudCollection Clean(RoadSearchCloudCollection *rawColl);

  SiStripMatchedRecHit2D*  CorrectMatchedHit(const TrackingRecHit* originalRH,
                                             const GluedGeomDet* gluedDet,
                                             const TrackerGeometry *tracker,
                                             const SiStripRecHitMatcher* theHitMatcher,
                                             double k0, double phi0);

 private:

  edm::ParameterSet conf_;
  static double epsilon;
  double d0h, phi0h, omegah;
  double rphicsq;
  int rphinhits;
  const SiPixelRecHitCollection thePixRecHits;
  
  // general hit access for road search
  DetHitAccess recHitVectorClass;
  
  double theRPhiRoadSize;
  double theZPhiRoadSize;
  double theMinimumHalfRoad;
  bool UsePixels;
  bool NoFieldCosmic;
  unsigned int maxDetHitsInCloudPerDetId;
  double       minFractionOfUsedLayersPerCloud;
  double       maxFractionOfMissedLayersPerCloud;
  double       maxFractionOfConsecutiveMissedLayersPerCloud;
  unsigned int increaseMaxNumberOfConsecutiveMissedLayersPerCloud;
  unsigned int increaseMaxNumberOfMissedLayersPerCloud;

  bool doCleaning_;
  double mergingFraction_;
  unsigned int maxRecHitsInCloud_;

  std::ostringstream output_;
  double scalefactorRoadSeedWindow_;

  std::string roadsLabel_;

};

#endif
