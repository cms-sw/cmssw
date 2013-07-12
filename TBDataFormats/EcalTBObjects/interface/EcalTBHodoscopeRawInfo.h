#ifndef DIGIECAL_ECALTBHODOSCOPERAWINFO_H
#define DIGIECAL_ECALTBHODOSCOPERAWINFO_H 1


#include <ostream>
#include <boost/cstdint.hpp>

/** \class EcalTBHodoscopeRawInfo
 *  Simple container for plane RawHits 
 *
 *
 *  $Id: EcalTBHodoscopeRawInfo.h,v 1.1 2006/03/28 13:51:36 meridian Exp $
 */
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"

class EcalTBHodoscopeRawInfo {
 public:
  EcalTBHodoscopeRawInfo() : planeHits_(0) {};

  EcalTBHodoscopeRawInfo(unsigned int planes) 
    {
      planeHits_.reserve(planes);
      for (unsigned int i=0;i<planes;i++)
	planeHits_[i]=0;
    }
  
  /// Get Methods
  unsigned int planes() const { return planeHits_.size(); } 
  unsigned int channels(unsigned int plane) const { return planeHits_[plane].channels(); }
  const std::vector<bool>& hits(unsigned int plane) const { return planeHits_[plane].hits(); }
  const EcalTBHodoscopePlaneRawHits& getPlaneRawHits(unsigned int i) const { return planeHits_[i]; }
  const EcalTBHodoscopePlaneRawHits& operator[](unsigned int i) const { return planeHits_[i]; }

  /// Set methods
  void setPlanes(unsigned int size)
    {
	planeHits_.resize(size);
    };
  
  void setPlane(unsigned int i, const EcalTBHodoscopePlaneRawHits& planeHit) 
    {
      if (planeHits_.size() < i+1 )
        planeHits_.resize(i+1);
      planeHits_[i]=planeHit; 
    };


 private:
  std::vector<EcalTBHodoscopePlaneRawHits> planeHits_;
  
};

std::ostream& operator<<(std::ostream&, const EcalTBHodoscopeRawInfo&);
  
#endif
