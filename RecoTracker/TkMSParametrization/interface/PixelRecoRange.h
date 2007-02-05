#ifndef PixelRecoRange_H
#define PixelRecoRange_H

/** Define a range [aMin,aMax] */

#include <iostream>
#include <utility>
#include <algorithm>

#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "RecoTracker/TkMSParametrization/interface/rangeIntersection.h"

template<class T> class PixelRecoRange : public std::pair<T,T> {
public:

  PixelRecoRange() { }

  PixelRecoRange(const T & aMin, const T & aMax) 
      : std::pair<T,T> (aMin,aMax) { }

  PixelRecoRange(const std::pair<T,T> & aPair ) 
      : std::pair<T,T> (aPair) { }  

  const T & min() const { return this->first; }
  const T & max() const { return this->second; }
  T mean() const { return (this->first+this->second)/2.; }

  bool empty() const { return (this->second < this->first); }

  bool inside(const T & value) const {
    if (value < this->first || this->second < value)  return false; else  return true;
  }

  bool hasIntersection( const PixelRecoRange<T> & r) const {
    return rangesIntersect(*this,r); 
  }

  PixelRecoRange<T> intersection( 
      const PixelRecoRange<T> & r) const {
    return rangeIntersection(*this,r); 
  }

  PixelRecoRange<T> sum(const PixelRecoRange<T> & r) const {
   if( this->empty()) return r;
   else if( r.empty()) return *this;
   else return 
       PixelRecoRange( 
         (min() < r.min()) ? min() : r.min(), 
         (max() < r.max()) ? r.max() : max()); 
  }

  void sort() { if (empty() ) std::swap(this->first,this->second); }
};

template <class T> std::ostream & operator<<( 
     std::ostream& out, const PixelRecoRange<T>& r) 
{
  return out << "("<<r.min()<<","<<r.max()<<")";
}
#endif
