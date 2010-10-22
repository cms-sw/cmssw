#ifndef CombinedKinematicConstraintT_H
#define CombinedKinematicConstraintT_H

#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraintT.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#if defined( __GXX_EXPERIMENTAL_CXX0X__)
// this is generic: to be moved elsewhere
#include<tuple>
#include<functional>
#include<algorithm>
#include<cassert>

// run time iteration
template<class TupleType, size_t N>
struct do_iterate 
{
  template<typename F>
  static void call(TupleType& t, F f) 
  {
    f(std::get<N-1>(t)); 
    do_iterate<TupleType, N-1>::call(t,f); 
  }
 template<typename F>
  static void call(TupleType const & t, F f) 
  {
    f(std::get<N-1>(t)); 
    do_iterate<TupleType, N-1>::call(t,f); 
  }


}; 

template<class TupleType>
struct do_iterate<TupleType, 0> 
{
  template<typename F>
  static void call(TupleType&, F) 
  {}
  template<typename F>
  static void call(TupleType const &, F) 
  {}
}; 

template<class TupleType, typename F>
void iterate_tuple(TupleType& t, F f)
{
  do_iterate<TupleType, std::tuple_size<TupleType>::value>::call(t,f);
}

template<class TupleType, typename F>
void iterate_tuple(TupleType const& t, F f)
{
  do_iterate<TupleType, std::tuple_size<TupleType>::value>::call(t,f);
}
 

namespace combinedConstraintHelpers {

  // a bit less generic
  template<class TupleType, size_t N=std::tuple_size<TupleType>::value>
  struct totDim {
    typedef typename std::tuple_element<N-1,TupleType>::type Elem;
    enum { nDim = Elem::nDim + totDim<TupleType,N-1>::nDim};
  };
  
  template<class TupleType>
  struct totDim<TupleType, 0>  {
    enum { nDim=0};
  };

  template<typename T>
  void sum2(T& x, T y) { x+=y;}

  // mind: iteration is backward...
  template<int DIM>
  struct Place {
    int offset;
    Place() : offset(DIM) {}
    ~Place() {
      assert(offset==DIM || offset==0);
    }
  };

  template<int DIM>
  struct PlaceValue : public Place<DIM> {
    ROOT::Math::SVector<double, DIM> ret;
    template<typename C>
    void operator()(C const & cs) {
      this->offset -= C::nDim;
      ret.Place_at(cs.value(),this->offset);
    }
  };

  template<int DIM, int NTRK>
  struct PlaceParDer : public Place<DIM> {
    ROOT::Math::SMatrix<double, DIM, 7*NTRK> ret;
    template<typename C>
    void operator()(C const & cs) {
      this->offset -= C::nDim;
      ret.Place_at(cs.parametersDerivative(),this->offset,0);
    }
  };

  template<int DIM>
  struct PlacePosDer : public Place<DIM> {
    ROOT::Math::SMatrix<double, DIM, 3> ret;
    template<typename C>
    void operator()(C const & cs) {
      this->offset -= C::nDim;
      ret.Place_at(cs.positionDerivative(),this->offset,0);
    }
  };


}




/**
 * This class combines several user defined constraints (by expanding the vector d and the matrices D and E).
 * Usage:
 * Add each constraint to a std::tuple<MultiTrackKinematicConstraint..... >.
 * This tuple has to be used in the constructor:
 *
 * The produced object can be used by KinematicConstrainedVertexFitter.fit()
 *
 * Lars Perchalla, Philip Sauerland, Dec 2009
 */

// maybe a variadic template will be better
template< class TupleType, int NTRK >
class CombinedKinematicConstraintT : public MultiTrackKinematicConstraintT<NTRK, combinedConstraintHelpers::totDim<TupleType>::nDim>{

  // need compile time assert on NTRK
public:
  typedef MultiTrackKinematicConstraintBaseT base;
  typedef MultiTrackKinematicConstraintT<NTRK,combinedConstraintHelpers::totDim<TupleType>::nDim> super;
  typedef typename super::valueType valueType;
  typedef typename super::parametersDerivativeType parametersDerivativeType;
  typedef typename super::positionDerivativeType positionDerivativeType;

  typedef TupleType Constraints;
  
  //FIXME
  enum {DIM = super::nDim};


public:
  CombinedKinematicConstraintT(Constraints const & iconstraints): constraints(constraints){
  }
  
  // initialize the constraint so it can precompute common qualtities to the three next call
  virtual void init(const std::vector<KinematicState>& states,
		    const GlobalPoint& point,  const GlobalVector& mf) {
    iterate_tuple(constraints,
		  std::bind(&base::init,std::placeholders::_1,std::ref(states),std::ref(point), std::ref(mf)));
  }


  /**
   * Returns a vector of values of the combined constraint
   * equations at the point where the input
   * particles are defined.
   */
  ROOT::Math::SVector<double, DIM>  value() const{
    combinedConstraintHelpers::PlaceValue<DIM> helper;
  } 
  
  /**
   * Returns a matrix of derivatives of the combined
   * constraint equations w.r.t. 
   * particle parameters
   */
  ROOT::Math::SMatrix<double, DIM, 7*NTRK> parametersDerivative() const{
    combinedConstraintHelpers::PlaceParDer<DIM,NTRK> helper;
    iterate_tuple(constraints,std::ref(helper));
    return helper.ret;
  }
  
  /**
   * Returns a matrix of derivatives of
   * constraint equations w.r.t. 
   * vertex position
   */
  virtual ROOT::Math::SMatrix<double, DIM, 3> positionDerivative() const{
    combinedConstraintHelpers::PlacePosDer<DIM> helper;
    iterate_tuple(constraints,std::ref(helper));
    return helper.ret;
  }
   
  /**
   * Number of equations per track used for the combined fit
   */
  virtual int numberOfEquations() const {
    int tot=0;
    iterate_tuple(constraints,std::bind(combinedConstraintHelpers::sum2<int>,std::ref(tot),
					std::bind(&base::numberOfEquations,std::placeholders::_1)
					)
		  );
    return tot;  				
  }
  
  virtual CombinedKinematicConstraintT * clone() const
  {
    return new CombinedKinematicConstraintT(*this);
  }
  
private:
  Constraints constraints;
  
};

#endif // __GXX_EXPERIMENTAL_CXX0X__

#endif
