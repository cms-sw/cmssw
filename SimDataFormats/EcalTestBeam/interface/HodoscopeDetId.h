#ifndef ECALDETID_HODOSCOPEDETID_H
#define ECALDETID_HODOSCOPEDETID_H

#include <ostream>
#include <cmath>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

/** \class HodoscopeDetId
 *  Hodoscope fiber identifier class for the ECAL TBH4 setup
 *
 *
 *  $Id: HodoscopeDetId.h,v 1.2 2011/06/03 19:17:00 heltsley Exp $
 */

// bkh June 2011: must be a calo detid type that is recognized by
//                CaloGenericDetId for use of its denseIndex() fcn.
//                Hence choose EBDetId to inherit from.

class HodoscopeDetId : public EBDetId 
{
   public:

      HodoscopeDetId();
      HodoscopeDetId( uint32_t rawid ) ;
      HodoscopeDetId( int iPlane, int iFibr ) ;
      HodoscopeDetId( const DetId& id ) ;

      int planeId() const ;

      int fibrId() const ;

      static bool validDetId( int iPlane , int iFibr ) ;

      /// range constants

      static const int MIN_PLANE =  0 ;
      static const int MAX_PLANE =  3 ;
      static const int MIN_FIBR  =  0 ;
      static const int MAX_FIBR  = 63 ;

};

std::ostream& operator<<(std::ostream& s,const HodoscopeDetId& id);


#endif
