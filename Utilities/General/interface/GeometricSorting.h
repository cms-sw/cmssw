#ifndef GeometricSorting_h
#define GeometricSorting_h

#include <functional>
#include <Geometry/Vector/interface/Phi.h>

namespace geomsort{

/** \class ExtractR
 *
 *  functor to sort in R using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractR());
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

  template <class T>
  struct ExtractR : public std::unary_function<const T, double> {
    double operator()(const T* p) const {return p->position().perp();}
    double operator()(const T& p) const {return p.position().perp();}
  };


/** \class ExtractPhi
 *
 *  functor to sort in phi (from -pi to pi) using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *
 *  Note that sorting in phi is done within the phi range of 
 *  (-pi, pi]. It may NOT be what you expect if the elements cluster around
 *  the pi discontinuity.
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractPhi());
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

  template <class T>
  struct ExtractPhi : public std::unary_function<const T, Geom::Phi<double> > {
    Geom::Phi<double> operator()(const T* p) const {return p->position().phi();}
    Geom::Phi<double> operator()(const T& p) const {return p.position().phi();}
  };



/** \class ExtractZ
 *
 *  functor to sort in Z using precomputed_value_sort.
 *  Can be used for any object with a member position(). 
 *  
 *  Use: 
 *
 *  precomputed_value_sort(v.begin(), v.end(), ExtractZ());
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

  template <class T>
  struct ExtractZ : public std::unary_function<const T, double> {
    double operator()(const T* p) const {return p->position().z();}
    double operator()(const T& p) const {return p.position().z();}
  };

};
#endif

