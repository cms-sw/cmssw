#ifndef ECALDETID_HODOSCOPEDETID_H
#define ECALDETID_HODOSCOPEDETID_H

#include <ostream>
#include <cmath>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

/** \class HodoscopeDetId
 *  Hodoscope fiber identifier class for the ECAL TBH4 setup
 *
 *
 *  $Id: HodoscopeDetId.h,v 1.1 2007/03/16 19:33:23 fabiocos Exp $
 */


class HodoscopeDetId : public DetId {
 public:
  /** Constructor of a null id */
  HodoscopeDetId();
  /** Constructor from a raw value */
  HodoscopeDetId(uint32_t rawid);
  /** Constructor from crystal ieta and iphi 
      or from SM# and crystal# */
  HodoscopeDetId(int indexPlane, int indexFibr);
  /** Constructor from a generic cell id */
  HodoscopeDetId(const DetId& id);
  /** Assignment operator from cell id */
  HodoscopeDetId& operator=(const DetId& id);

  /// get the subdetector
  EcalSubdetector subdet() const { return EcalSubdetector(subdetId()); }

  int planeId() const { return id_&0x3 ; }

  int fibrId() const { return (id_>>2)&0x3F ; }

  /// range constants
  static const int MIN_PLANE = 0;
  static const int MAX_PLANE = 3;
  static const int MIN_FIBR = 0;
  static const int MAX_FIBR = 63;
    
};

std::ostream& operator<<(std::ostream& s,const HodoscopeDetId& id);


#endif
