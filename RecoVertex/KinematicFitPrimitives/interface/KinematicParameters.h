#ifndef KinematicParameters_H
#define KinematicParameters_H

#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/**
 * Class to store the 7-vector of
 * particle parameters: (x,y,z,p_x,p_y,p_z,m)
 *
 * Kirill Prokofiev Febrauary 2003
 */


class KinematicParameters{

public:

  typedef ROOT::Math::SVector<double,7> AlgebraicVector7;

  KinematicParameters() : vl(false) {}

  KinematicParameters(const AlgebraicVector7& pr): par(pr),vl(true)
  {}

  /**
   * \brief The full vector (7 elements)
   * 
   * The order of the parameters is (x,y,z,p_x,p_y,p_z,m)
   */
  AlgebraicVector7 const & vector() const {return par;}

  /**
   * \brief Allows to access directly one component of the vector (index between 0 and 6)
   * 
   * The order of the parameters is (x,y,z,p_x,p_y,p_z,m)
   */
  double operator()(const int i) const  {return par(i);}

  /**
   * The momentum vector
   */
  GlobalVector momentum() const;
  
  /**
   * The position of the state
   */
  GlobalPoint position() const;
  
  /**
   * The mass of the particle
   */
  double mass() const {return par(6);}

  /**
   * The energy of the particle
   */
  double energy() const;

  bool isValid() const
  {return vl;}

private:
   AlgebraicVector7 par;
   bool vl;
};


#endif
