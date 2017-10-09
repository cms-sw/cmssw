#ifndef MultiTrackKinematicConstraintT_H
#define MultiTrackKinematicConstraintT_H

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


/**
 * Pure abstract class implementing constraint application 
 * on multiple tracks (back to back, collinearity etc.)
 * To be used by KinematicConstraindeVertexFitter only
 * Class caches the information about calculation of
 * of constraint equation derivatives and values at given
 * linearization point. Point should be of 7*n+3 dimensions
 * Where n - number of particles. 7 - parametrization for 
 * particles is (x,y,z,p_x,p_y,p_z,m), for vertex (x_v,y_v,z_v)
 * Fitter usually takes current parameters as the first step point
 * and the change it to the result of the first iteration.
 *
 * Kirill Prokofiev, October 2003
 *
 * New version: make use of SMatrix: optimize filling of matrices
 * Vincenzo Innocente, October 2010
 */

class MultiTrackKinematicConstraintBaseT
{
public:
  virtual ~MultiTrackKinematicConstraintBaseT() {}

 // initialize the constraint so it can precompute common qualtities to the three next call
  virtual void init(const std::vector<KinematicState>& states,
		    const GlobalPoint& point,  const GlobalVector& mf) =0;

  virtual int numberOfEquations() const = 0;
  
  virtual MultiTrackKinematicConstraintBaseT * clone() const = 0;


};

template<int NTRK, int DIM>
class MultiTrackKinematicConstraintT : public MultiTrackKinematicConstraintBaseT
{
public:
  enum {nTrk=NTRK, nDim=DIM};

  typedef MultiTrackKinematicConstraintT<NTRK, DIM> self;

  typedef ROOT::Math::SVector<double, DIM>  valueType; 
  
  typedef ROOT::Math::SMatrix<double, DIM,7*NTRK> parametersDerivativeType;
  

  typedef ROOT::Math::SMatrix<double, DIM,3> positionDerivativeType;


  virtual ~MultiTrackKinematicConstraintT() {}


  /**
   * Methods returning vector of values
   * and derivative matrices with
   * respect to vertex position and
   * particle parameters.
   * Input paramters are put into one vector: 
   * (Vertex position, particle_parameters_1,..., particle_parameters_n)
   */
  valueType const & value() const {
    fillValue();
    return m_vl;
  } 
  
  parametersDerivativeType const & parametersDerivative() const {
    fillParametersDerivative();
    return m_jac_d;
  };
  

  positionDerivativeType const &  positionDerivative() const {
    fillPositionDerivative();
    return m_jac_e;
  }
 
private:
  /**
   * Methods making vector of values
   * and derivative matrices with
   * respect to vertex position and
   * particle parameters.
   * Input paramters are put into one vector: 
   * (Vertex position, particle_parameters_1,..., particle_parameters_n)
   */
  virtual  void fillValue() const = 0; 
  
  virtual  void fillParametersDerivative() const = 0;
  

  virtual void fillPositionDerivative() const = 0;
  
protected:

  valueType & vl() const { return m_vl; }
  parametersDerivativeType & jac_d() const { return m_jac_d;}
  positionDerivativeType & jac_e() const { return m_jac_e;}

  // self & me() const { return *const_cast<self*>(this); }

  double & vl(size_t i) const { return m_vl(i);}
  double & jac_d(size_t i, size_t j) const { return m_jac_d(i,j);}
  double & jac_e(size_t i, size_t j) const { return m_jac_e(i,j);}

private:

  mutable valueType m_vl;
  mutable parametersDerivativeType m_jac_d;
  mutable positionDerivativeType m_jac_e;

};


#endif
