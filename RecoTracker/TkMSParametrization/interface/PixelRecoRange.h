#ifndef PixelRecoRange_H
#define PixelRecoRange_H

/** Define a range [aMin,aMax] */

#include <ostream>
#include <utility>
#include <algorithm>

#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "RecoTracker/TkMSParametrization/interface/rangeIntersection.h"

template<class T> class PixelRecoRange : public std::pair<T,T> {
public:

  PixelRecoRange() { }


  template<class U>
  PixelRecoRange(PixelRecoRange<U> other) : 
  std::pair<T,T> (other.min(),other.max()) { }


  PixelRecoRange(T  aMin, T aMax) 
      : std::pair<T,T> (aMin,aMax) { }

  PixelRecoRange(const std::pair<T,T> & aPair ) 
      : std::pair<T,T> (aPair) { }  

  T min() const { return this->first; }
  T max() const { return this->second; }
  T mean() const { return T(0.5)*(this->first+this->second); }

  bool empty() const { return (this->second < this->first); }

  PixelRecoRange<T> & swap() { std::swap(this->second,this->first); return *this;}


  bool inside(const T & value) const {
    return !( (value < this->first) | (this->second < value) );
  }

  bool hasIntersection( const PixelRecoRange<T> & r) const {
    return rangesIntersect(*this,r); 
  }

  PixelRecoRange<T> intersection( const PixelRecoRange<T> & r) const {
    return rangeIntersection(*this,r); 
  }

  PixelRecoRange<T> sum(const PixelRecoRange<T> & r) const {
   if( this->empty()) return r;
   else if( r.empty()) return *this;
   else return 
	  PixelRecoRange(std::min( min(),r.min()),
			 std::max( max(),r.max())
                        ); 
  }

  PixelRecoRange<T> & sort() { if (empty() ) std::swap(this->first,this->second); return *this;}

};

template <class T> std::ostream & operator<<( 
     std::ostream& out, const PixelRecoRange<T>& r) 
{
  return out << "("<<r.min()<<","<<r.max()<<")";
}
#endif
