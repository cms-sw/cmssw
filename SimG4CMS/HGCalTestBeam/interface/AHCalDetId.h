#ifndef SimG4CMS_HGCalTestBeam_HCALDETID_H
#define SimG4CMS_HGCalTestBeam_HCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


/** \class AHCalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells only
 */
class AHCalDetId : public DetId {
public:
  /** Create a null cellid*/
  AHCalDetId();
  /** Create cellid from raw id (0=invalid tower id) */
  AHCalDetId(uint32_t rawid);
  /** Constructor from subdetector, signed tower ieta,iphi,and depth */
  AHCalDetId(int row, int col, int depth);
  /** Constructor from a generic cell id */
  AHCalDetId(const DetId& id);
  /** Assignment from a generic cell id */
  AHCalDetId& operator=(const DetId& id) {
    id_=id.rawId();
  return *this;
  }

  /// get the subdetector
  HcalSubdetector subdet() const { return HcalOther; }
  /// get the z-side of the cell (1/-1)
  int zside() const { return 1; }
  /// get the row number
  int irow() const;
  int irowAbs() const { 
    return ((id_>>HcalDetId::kHcalEtaOffset2)&HcalDetId::kHcalEtaMask2); }
  /// get the column number
  int icol() const;
  int icolAbs() const { return (id_&HcalDetId::kHcalPhiMask2); }
  /// get the layer number
  static const int MaxDepth=12;
  int depth() const;
  /// get the local coordinate in the plane and along depth
  std::pair<double,double> getXY() const;
  double getZ() const;

  static const AHCalDetId Undefined;
  const double deltaX_ = 3.0;  // Size of tile along X
  const double deltaY_ = 3.0;  // Size of tile along Y
  const double deltaZ_ = 8.1;  // Thickness of a single layer
  const double zFirst_ = 1.76; // Position of the center 
};

std::ostream& operator<<(std::ostream&,const AHCalDetId& id);

#endif
