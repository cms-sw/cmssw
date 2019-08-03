#ifndef SimG4CMS_HGCalTestBeam_AHCALDETID_H
#define SimG4CMS_HGCalTestBeam_AHCALDETID_H 1

#include <iosfwd>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

/** \class AHCalDetId
 *  Cell identifier class for the HCAL subdetectors, precision readout cells
 * only
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
    id_ = id.rawId();
    return *this;
  }

  /// get the subdetector
  HcalSubdetector subdet() const { return HcalOther; }
  /// get the z-side of the cell (1/-1)
  int zside() const { return 1; }
  /// get the row number
  int irow() const;
  int irowAbs() const { return ((id_ >> HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2); }
  /// get the column number
  int icol() const;
  int icolAbs() const { return (id_ & HcalDetId::kHcalPhiMask2); }
  /// get the layer number
  int depth() const;

  static const AHCalDetId Undefined;
};

std::ostream& operator<<(std::ostream&, const AHCalDetId& id);

#endif
