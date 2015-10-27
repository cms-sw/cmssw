#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
HodoscopeDetId::HodoscopeDetId() : 
   EBDetId() 
{
}

HodoscopeDetId::HodoscopeDetId( uint32_t rawid ) : 
   EBDetId( rawid ) 
{
}

int 
HodoscopeDetId::planeId() const 
{
   return ieta() ; 
}

int 
HodoscopeDetId::fibrId() const 
{ 
   return iphi() ; 
}

HodoscopeDetId::HodoscopeDetId( int iPlane ,
				int iFibr    ) 
   : EBDetId( iPlane, iFibr )
{
   if( !validDetId( iPlane, iFibr ) )
   {
      throw cms::Exception("InvalidDetId") 
	 << "HodoscopeDetId:  Cannot create object.  Indices out of bounds.";
   }
}
  
HodoscopeDetId::HodoscopeDetId( const DetId& gen ) :
   EBDetId( gen )
{
   if( !validDetId( planeId(), fibrId() ) )
   {
      throw cms::Exception("InvalidDetId") 
	 << "HodoscopeDetId:  Cannot create object.  Indices out of bounds.";
   }
}

bool 
HodoscopeDetId::validDetId( int iPlane ,
			    int iFibr   ) 
{
   return !( iPlane < MIN_PLANE || 
	     iPlane > MAX_PLANE ||
	     iFibr  < MIN_FIBR  ||
	     iFibr  > MAX_FIBR     ) ;
}

std::ostream& operator<<(std::ostream& s,const HodoscopeDetId& id) 
{
   return s << "(Plane " << id.planeId() << ", fiber " << id.fibrId() << ')';
}
  
