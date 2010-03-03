#ifndef TRACKERTRIGGERNAIVEGEOMETRY
#define TRACKERTRIGGERNAIVEGEOMETRY

/// Naive tracker geometry
/// J Brooke, Mar 08
///
/// Defines tracker layers and module geometry within each layer
///
/// tower ID bit : barrel  : endcap
/// 31           : sgn(z)  : sgn(z)
/// 30           : 0       : 1
/// 29..27       : layer   : disk
/// 26..14       : iz      : ir
/// 13..0        : iphi    : iphi


#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerMath.h"

#include <vector>

class TrackTriggerNaiveGeometry {

 public:
  TrackTriggerNaiveGeometry();
  TrackTriggerNaiveGeometry(std::vector<double> barrelRadii, 
		  					std::vector<double> barrelLengths, 
		  					std::vector<double> barrelTowZSize,
		  					std::vector<double> barrelTowPhiSize,
//		  					std::vector<double> diskZPos,
//		  					std::vector<double> diskRadii, 
//		  					std::vector<double> diskTowPhiSize,
//		  					std::vector<double> diskTowRSize,
		  					double barrelPixelZSize,
		  					double barrelPixelPhiSize);
//		  					double diskPixelPhiZSize,
//		  					double diskPixelRSize);

  // basic geometry info

  /// number of barrel layers
  unsigned nLayers() const;

  /// barrel layer radii (cm)
  double barrelLayerRadius(int l) const;

  /// barrel layer length (cm)
  double barrelLayerLength(int l) const;

//  /// number of endcap disks (in one endcap!)
//  unsigned nDisks() const;
//
//  /// endcap disk z position (cm)
//  double diskZPos(int disk) const;
//
//  /// endcap disk radius (cm)
//  double diskRadius(int disk) const;


  // tower info
  
  /// tower size in Z in barrel (cm)
  double towerSizeZ(unsigned layer) const;

  /// tower size in phi in barrel (cm!)
  double towerSizePhi(unsigned layer) const;
  
  /// number of towers in barrel in z
  unsigned nBarrelTowersZ(unsigned layer) const;

  /// number of towers in barrel in phi
  unsigned nBarrelTowersPhi(unsigned layer) const;	

  /// tower sign of z
  int towerZSign(unsigned id) const;
  
  /// is tower in the barrel?
  bool towerIsBarrel(unsigned id) const;
  
  /// barrel tower layer
  unsigned barrelTowerILayer(unsigned id) const;

  /// barrel tower z index (for a half-barrel!)
  unsigned barrelTowerIZ(unsigned id) const;

  /// barrel tower z index
  unsigned barrelTowerIPhi(unsigned id) const;

  /// barrel tower centre z position (for a half barrel!) (cm)
  double barrelTowerLowZ(unsigned ilayer, unsigned iz) const;

  /// barrel tower centre phi position (radians)
  double barrelTowerLowPhi(unsigned ilayer, unsigned iphi) const;

  /// barrel tower centre z position (for a half barrel!) (cm)
  double barrelTowerCentreZ(unsigned ilayer, unsigned iz) const;

  /// barrel tower centre phi position (radians)
  double barrelTowerCentrePhi(unsigned ilayer, unsigned iphi) const;

  /// barrel tower id from position (z/cm, phi/radians)
  unsigned barrelTowerId(unsigned layer, double z, double phi) const;

//  /// endcap tower disk
//  unsigned diskTowerIDisk(unsigned id) const;
//
//  /// endcap tower r index
//  int diskTowerIR(unsigned id) const;
//
//  /// endcap tower z index
//  int diskTowerIPhi(unsigned id) const;
//
//  /// endcap tower id from position
//  unsigned diskTowerId(unsigned disk, double r, double phi) const;

  
  // pixel info
  /// pixel size in z (cm)
  double barrelPixelSizeZ(unsigned ilayer) const;
		
  /// pixel size in phi (cm!)
  double barrelPixelSizePhi(unsigned ilayer) const;

  /// n pixels per tower in z
  unsigned nBarrelPixelsZ(unsigned ilayer) const;
		
  /// n pixels per tower in phi
  unsigned nBarrelPixelsPhi(unsigned ilayer) const;

  /// get z coord from row in barrel (cm)
  double barrelPixelZ(unsigned towId, unsigned row) const;

  /// get phi coord from col in barrel (radians)
  double barrelPixelPhi(unsigned towId, unsigned col) const;

  /// get row from z coordinate in barrel
  unsigned barrelPixelRow(unsigned towId, double z) const; 

  /// get column from phi coordinate in barrel
  unsigned barrelPixelColumn(unsigned towId, double phi) const; 

//  /// get r coord from row in endcap
//  double endcapPixelR(unsigned towId, unsigned row) const;
//
//  /// get phi coord from col in endcap
//  double endcapPixelPhi(unsigned towId, unsigned col) const;
//
//  /// get row from r coordinate in endcap
//  unsigned endcapPixelRow(unsigned towId, double r) const; 
//
//  /// get column from phi coordinate in barrel
//  unsigned endcapPixelColumn(unsigned towId, double phi) const; 

  friend std::ostream& operator << (std::ostream& os, const TrackTriggerNaiveGeometry& hit);  

 private:

  // barrel
	 unsigned nLayers_;
	 std::vector<double> barrelRadii_;
	 std::vector<double> barrelLengths_;
	 std::vector<double> barrelTowZSize_;
	 std::vector<double> barrelTowPhiSize_;
	 std::vector<unsigned> nTowersZ_;
	 std::vector<unsigned> nTowersPhi_;
	 double barrelPixelZSize_;
	 double barrelPixelPhiSize_;

  // endcap
//  std::vector<double> diskZPos_;
//  std::vector<double> diskRadii_;
//  std::vector<double> diskTowPhiSize_;
//  std::vector<double> diskTowRSize_;
//  double diskPixelRSize_;
//  double diskPixelPhiSize_;

  
};


#endif

