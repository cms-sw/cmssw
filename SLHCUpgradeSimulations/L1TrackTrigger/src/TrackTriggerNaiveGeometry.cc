#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerNaiveGeometry.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <TMath.h>
#include <cmath>
#include <stdexcept>

/// tower ID bit : barrel  : endcap
/// 31           : sgn(z)  : sgn(z)  (0=+ve, 1=-ve)
/// 30           : 0       : 1
/// 29..27       : layer   : disk
/// 26..14       : iz      : ir
/// 13..0        : iphi    : iphi


TrackTriggerNaiveGeometry::TrackTriggerNaiveGeometry(std::vector<double> barrelRadii, 
								std::vector<double> barrelLengths, 
								std::vector<double> barrelTowZSize,
								std::vector<double> barrelTowPhiSize,
//								std::vector<double> diskZPos,
//								std::vector<double> diskRadii, 
//								std::vector<double> diskTowPhiSize,
//								std::vector<double> diskTowRSize,
								double barrelPixelZSize,
								double barrelPixelPhiSize) :
//								double diskPixelPhiSize,
//								double diskPixelRZSize) :
	nLayers_(barrelRadii.size()),	
	barrelRadii_(barrelRadii),
	barrelLengths_(barrelLengths),
	barrelTowZSize_(barrelTowZSize),
	barrelTowPhiSize_(barrelTowPhiSize),
//  diskZPos_(diskZPos),
//  diskRadii_(diskRadii),
//  diskTowPhiBoundaries_(diskTowPhiBoundaries),
//  diskTowRBoundaries_(diskTowRBoundaries),
	barrelPixelZSize_(barrelPixelZSize),
	barrelPixelPhiSize_(barrelPixelPhiSize)
//  diskPixelRSize_(diskPixelRZSize),
//  diskPixelPhiSize_(diskPixelPhiSize)
{ 

	// check vectors are the right size
	if(barrelLengths_.size() != nLayers_ ||
		barrelTowPhiSize_.size() != nLayers_ ||
		barrelTowZSize_.size() != nLayers_) 
	{
		throw cms::Exception("InvalidParameter")
		<< "Can't construct TrackTriggerNaiveGeometry from vectors of different sizes!" << std::endl;
	}
	
}


/// number of barrel layers
unsigned TrackTriggerNaiveGeometry::nLayers() const {
	return barrelRadii_.size(); 
}

/// barrel layer radii (cm)
double TrackTriggerNaiveGeometry::barrelLayerRadius(int il) const {
	try { return barrelRadii_.at(il); }
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "TrackTriggerNaiveGeometry::barrelLayerRadius(" << il << ") : layer index out of range" << std::endl;
	}
}

/// barrel layer length (cm)
double TrackTriggerNaiveGeometry::barrelLayerLength(int il) const {
	try { return barrelLengths_.at(il); }
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "TrackTriggerNaiveGeometry::barrelLayerLength(" << il << ") : layer index out of range" << std::endl;
	}
}

///// number of endcap disks (in one endcap!)
//unsigned TrackTriggerNaiveGeometry::nDisks() const {
//	return diskZPos_.size(); 
//}
//
///// endcap disk z position (cm)
//double TrackTriggerNaiveGeometry::diskZPos(int disk) const {
//	try { return diskZPos_.at(disk); }
//	catch (std::out_of_range) {
//		throw cms::Exception("OutOfRange")
//		<< "TrackTriggerNaiveGeometry::diskZPos" << disk << ") : disk index out of range" << std::endl;	
//	}
//}
//
///// endcap disk radius (cm)
//double TrackTriggerNaiveGeometry::diskRadius(int disk) const {
//	try { return diskRadii_.at(disk); }
//	catch (std::out_of_range) {
//		throw cms::Exception("OutOfRange")
//		<< "TrackTriggerNaiveGeometry::diskZPos" << disk << ") : disk index out of range" << std::endl;	
//	}
//}



// tower info methods

/// tower size in Z in barrel
double TrackTriggerNaiveGeometry::towerSizeZ(unsigned il) const {
	try {
		return barrelTowZSize_.at(il);
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}

/// tower size in phi in barrel
double TrackTriggerNaiveGeometry::towerSizePhi(unsigned il) const {
	try {
		return barrelTowPhiSize_.at(il);
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}

/// number of towers in z in a half-barrel
unsigned TrackTriggerNaiveGeometry::nBarrelTowersZ(unsigned il) const {
	try {
		return (unsigned) std::ceil(barrelLengths_.at(il)/barrelTowPhiSize_.at(il));
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}

/// number of towers in phi in barrel
unsigned TrackTriggerNaiveGeometry::nBarrelTowersPhi(unsigned il) const {
	try {
		return (unsigned) std::ceil(2*TMath::Pi()*barrelRadii_.at(il)/barrelTowPhiSize_.at(il));
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}

/// sign of z
int TrackTriggerNaiveGeometry::towerZSign(unsigned id) const {
  return ( (id>>31)&0x1 == 1 ? -1 : +1 ) ;
}

/// is tower in the barrel?
bool TrackTriggerNaiveGeometry::towerIsBarrel(unsigned id) const {
  return !((id>>30)&0x1 == 1);
}

/// barrel tower layer
unsigned TrackTriggerNaiveGeometry::barrelTowerILayer(unsigned id) const {
	unsigned l = (id>>27)&0x7;
	return ( l<nLayers_ ? l : 0) ;
}

/// barrel tower z index (for a half-barrel!)
unsigned TrackTriggerNaiveGeometry::barrelTowerIZ(unsigned id) const  {
	if (!towerIsBarrel(id)) return 0;
	unsigned iz = ((id>>14) & 0x1fff);
	unsigned il = barrelTowerILayer(id);
	if (iz < nBarrelTowersZ(il)) return iz;
	else throw cms::Exception("InvalidID") << "TrackTriggerNaiveGeometry::barrelTowerIZ() : "
	<< " ID" << id << " has z index " << iz << std::endl;
}

/// barrel tower phi index
unsigned TrackTriggerNaiveGeometry::barrelTowerIPhi(unsigned id) const {
	if (!towerIsBarrel(id)) return 0;
	unsigned iphi = id & 0x1fff;
	unsigned il = barrelTowerILayer(id);
	if (iphi < nBarrelTowersPhi(il)) return iphi;
	else throw cms::Exception("InvalidID") << "TrackTriggerNaiveGeometry::barrelTowerIPhi() : "
	<< " ID" << id << " has phi index " << iphi << std::endl;
}

/// barrel tower centre position z (for a half-barrel!)
double TrackTriggerNaiveGeometry::barrelTowerLowZ(unsigned il, unsigned iz) const {
	if (il >= nLayers_ ) {
		throw cms::Exception("OutOfRange") << "layer index " << il << " out of range" << std::endl;	
	}
	if (iz < nBarrelTowersZ(il)) return iz * towerSizeZ(il);
	else throw cms::Exception("OutOfRange") << "z index " << iz << " out of range" << std::endl;	
}

/// barrel tower centre position phi (radians)
double TrackTriggerNaiveGeometry::barrelTowerLowPhi(unsigned il, unsigned iphi) const {
	if (il >= nLayers_ ) {
		throw cms::Exception("OutOfRange") << "layer index " << il << " out of range" << std::endl;	
	}
	if (iphi < nBarrelTowersPhi(il)) return (-1*TMath::Pi()) + (iphi * towerSizePhi(il)/barrelLayerRadius(il));
	else throw cms::Exception("OutOfRange") << "phi index " << iphi << " out of range" << std::endl;	
}

/// barrel tower centre position z (for a half-barrel!)
double TrackTriggerNaiveGeometry::barrelTowerCentreZ(unsigned il, unsigned iz) const {
	if (il >= nLayers_ ) {
		throw cms::Exception("OutOfRange") << "layer index " << il << " out of range" << std::endl;	
	}
	if (iz < nBarrelTowersZ(il)) return (iz + 0.5) * towerSizeZ(il);
	else throw cms::Exception("OutOfRange") << "z index " << iz << " out of range" << std::endl;	
}

/// barrel tower centre position phi (radians)
double TrackTriggerNaiveGeometry::barrelTowerCentrePhi(unsigned il, unsigned iphi) const {
	if (il >= nLayers_ ) {
		throw cms::Exception("OutOfRange") << "layer index " << il << " out of range" << std::endl;	
	}
	if (iphi < nBarrelTowersPhi(il)) return (-1*TMath::Pi()) + ((iphi + 0.5) * towerSizePhi(il)/barrelLayerRadius(il));
	else throw cms::Exception("OutOfRange") << "phi index " << iphi << " out of range" << std::endl;	
}

/// barrel tower id from position (_full_ barrel)
unsigned TrackTriggerNaiveGeometry::barrelTowerId(unsigned il, double z, double phi) const {
	unsigned zsgn = (z>0 ? 0 : 1);
	unsigned iz, iphi;
	for (iz=0; iz<nBarrelTowersZ(il) &&
		fabs(z) > barrelTowerLowZ(il, iz); ++iz) { 
	}
	--iz;
	for (iphi=0; iphi<nBarrelTowersPhi(il) && 
		phi > barrelTowerLowPhi(il, iphi); ++iphi) { 
	}
	--iphi;
	if (il >= nLayers() || iz >= nBarrelTowersZ(il) || iphi >= nBarrelTowersPhi(il)) {
		throw cms::Exception("InvalidID") << "Generated Invalid ID! :"
		<< " z=" << z << " phi=" << phi
		<< " il=" << il << " iz=" << iz << " iphi=" << iphi << std::endl;
	}
	return ((zsgn&0x1)<<31) + ((il&0x7)<<27) + ((iz&0x1fff)<<14) + (iphi&0x1fff);
}

///// endcap tower disk
//unsigned TrackTriggerNaiveGeometry::diskTowerIDisk(unsigned id) const {
//  return ( towerIsBarrel(id) ? 0 : (id>>27)&0x7 );
//}
//
///// endcap tower r index
//int TrackTriggerNaiveGeometry::diskTowerIR(unsigned id) const {
//  return ( towerIsBarrel(id) ? 0 : (id>>14) & 0x1fff );
//}
//
///// endcap tower z index
//int TrackTriggerNaiveGeometry::diskTowerIPhi(unsigned id) const {
//  return ( towerIsBarrel(id) ? 0 : id & 0x1fff );
//}
//
///// endcap tower id from position
//unsigned TrackTriggerNaiveGeometry::diskTowerId(unsigned disk, double r, double phi) const {
//// TODO - endcaps
//  return 0;
//}


// pixel info
/// pixel size in z
double TrackTriggerNaiveGeometry::barrelPixelSizeZ(unsigned il) const {
	return barrelPixelZSize_;
}
		
/// pixel size in phi
double TrackTriggerNaiveGeometry::barrelPixelSizePhi(unsigned il) const {
	return barrelPixelPhiSize_;
}

/// pixel size in z
unsigned TrackTriggerNaiveGeometry::nBarrelPixelsZ(unsigned il) const {
	try {
		return (unsigned) std::ceil(barrelTowZSize_.at(il)/barrelPixelZSize_);
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}
		
/// pixel size in phi
unsigned TrackTriggerNaiveGeometry::nBarrelPixelsPhi(unsigned il) const {
	try {
		return (unsigned) std::ceil(barrelTowPhiSize_.at(il)/barrelPixelPhiSize_);
	}
	catch (std::out_of_range) {
		throw cms::Exception("OutOfRange")
		<< "layer index " << il << " out of range" << std::endl;
	}
}


/// get z coord from row in barrel
double TrackTriggerNaiveGeometry::barrelPixelZ(unsigned towId, unsigned row) const {
	return (towerZSign(towId) * (towerSizeZ(barrelTowerILayer(towId))) + (row * barrelPixelZSize_));
}

/// get phi coord from col in barrel
double TrackTriggerNaiveGeometry::barrelPixelPhi(unsigned towId, unsigned col) const {
	unsigned il=barrelTowerILayer(towId);
	return (towerSizePhi(il) + (col * barrelPixelPhiSize_))/barrelLayerRadius(il);
}

/// get row from z coordinate in barrel
unsigned TrackTriggerNaiveGeometry::barrelPixelRow(unsigned towId, double z) const {
	unsigned il = barrelTowerILayer(towId);
	unsigned iz = barrelTowerIZ(towId);
	double towLoZ = barrelTowerLowZ(il, iz);
	unsigned row=0;
	for (row=0; row<=nBarrelPixelsZ(il) &&
			towLoZ+(row*barrelPixelZSize_) < fabs(z); ++row) { }
	return row-1;
}

/// get column from phi coordinate in barrel
unsigned TrackTriggerNaiveGeometry::barrelPixelColumn(unsigned towId, double phi) const {
	unsigned il = barrelTowerILayer(towId);
	unsigned iphi = barrelTowerIPhi(towId);
	double towLoPhi = barrelTowerLowPhi(il, iphi);
	unsigned col=0;
	for (col=0; col<=nBarrelPixelsPhi(il) &&
			towLoPhi+(col*barrelPixelPhiSize_/barrelLayerRadius(il)) < phi; ++col) { }
	return col-1;
}


///// get r coord from row in endcap
//double TrackTriggerNaiveGeometry::endcapPixelR(unsigned towId, unsigned row) const {
//  // TODO - write pixel methods
//  	return 0.;
//}
//
///// get phi coord from col in endcap
//double TrackTriggerNaiveGeometry::endcapPixelPhi(unsigned towId, unsigned col) const {
//  // TODO - write pixel methods
//  	return 0.;
//}
//
///// get row from r coordinate in endcap
//unsigned TrackTriggerNaiveGeometry::endcapPixelRow(unsigned towId, double r) const {
//  // TODO - write pixel methods
//  return 0;
//}
//
///// get column from phi coordinate in barrel
//unsigned TrackTriggerNaiveGeometry::endcapPixelColumn(unsigned towId, double phi) const { 
//  // TODO - write pixel methods
//  return 0;
//}

std::ostream& operator << (std::ostream& os, const TrackTriggerNaiveGeometry& g) {
  os << "TrackTriggerNaiveGeometry :";
  os << " nLayers=" << g.nLayers();
  os << " nTowersZ=" << g.nBarrelTowersZ(0);
  os << " nTowersPhi=" << g.nBarrelTowersPhi(0);
  os << " towerSizeZ=" << g.towerSizeZ(0);
  os << " towerSizePhi=" << g.towerSizePhi(0);
  os << " nPixelsZ=" << g.nBarrelPixelsZ(0);
  os << " nPixelsPhi=" << g.nBarrelPixelsPhi(0);
  os << " pixelSizeZ=" << g.barrelPixelSizeZ(0);
  os << " pixelSizePhi=" << g.barrelPixelSizePhi(0);
  return os;
}

