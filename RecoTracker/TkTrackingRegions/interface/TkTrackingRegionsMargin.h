#ifndef TkTrackingRegionsMargin_H
#define TkTrackingRegionsMargin_H

/** Define two (left and right) margins */

#include <utility>
#include <iostream>

template <class T> class TkTrackingRegionsMargin : public pair<T,T> {
public:

  TkTrackingRegionsMargin() { }

  TkTrackingRegionsMargin(const T & aLeft, const T & aRight) 
    : pair<T,T> (aLeft,aRight) { }

  TkTrackingRegionsMargin( const pair<T,T> & aPair)
    : pair<T,T> (aPair) { }

  const T & left() const { return this->first; }
  const T & right() const { return this->second; }

  void operator += ( const T & v) { add(v,v); }
  void add( const T & addLeft, const T & addRight) { 
    this->first += addLeft; 
    this->second += addRight; 
  } 
};

template <class T> ostream & operator<< ( 
    ostream& out, const TkTrackingRegionsMargin<T> & m) 
{
  return out  << "("<<m.left()<<","<<m.right()<<")";
}
#endif
  
