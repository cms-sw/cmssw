#ifndef MaterialAccountingStep_h
#define MaterialAccountingStep_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// struct to keep material accounting informations on a per-step basis
// TODO split segment info (in, out) into separate child class
class MaterialAccountingStep {
public:
  MaterialAccountingStep( void ) :
    m_length(0.),
    m_radiationLengths(0.),
    m_energyLoss(0.),
    m_in(),
    m_out() 
  { }

  MaterialAccountingStep( double position, double radlen, double loss, const GlobalPoint & in, const GlobalPoint & out ) :
    m_length( position ),
    m_radiationLengths( radlen ),
    m_energyLoss( loss ),
    m_in( in ),
    m_out( out )
  { }
 
  void clear( void ) {
    m_length           = 0.;
    m_radiationLengths = 0.;
    m_energyLoss       = 0.;
    m_in               = GlobalPoint();
    m_out              = GlobalPoint();
  }

private:  
  double m_length;
  double m_radiationLengths;
  double m_energyLoss;
  GlobalPoint m_in;
  GlobalPoint m_out;

public:
  double length(void) const {
    return m_length;
  }
  
  double radiationLengths(void) const {
    return m_radiationLengths;
  }
  
  double energyLoss(void) const {
    return m_energyLoss;
  }

  const GlobalPoint & in(void) const {
    return m_in;
  }
  
  const GlobalPoint & out(void) const {
    return m_out;
  }
  
  /// assignement operator
  MaterialAccountingStep & operator=( const MaterialAccountingStep & step ) {
    m_length           = step.m_length;
    m_radiationLengths = step.m_radiationLengths;
    m_energyLoss       = step.m_energyLoss;
    return *this;
  }
  
  /// add a step
  MaterialAccountingStep & operator+=( const MaterialAccountingStep & step ) {
    m_length           += step.m_length;
    m_radiationLengths += step.m_radiationLengths;
    m_energyLoss       += step.m_energyLoss;
    return *this;
  }
  
  /// subtract a step
  MaterialAccountingStep & operator-=( const MaterialAccountingStep & step ) {
    m_length           -= step.m_length;
    m_radiationLengths -= step.m_radiationLengths;
    m_energyLoss       -= step.m_energyLoss;
    return *this;
  }
  
  /// multiply two steps, usefull to implement (co)variance
  MaterialAccountingStep & operator*=( const MaterialAccountingStep & step ) {
    m_length           *= step.m_length;
    m_radiationLengths *= step.m_radiationLengths;
    m_energyLoss       *= step.m_energyLoss;
    return *this;
  }
  
  /// multiply by a scalar
  MaterialAccountingStep & operator*=( double x ) {
    m_length           *= x;
    m_radiationLengths *= x;
    m_energyLoss       *= x;
    return *this;
  }
  
  /// divide by a scalar
  MaterialAccountingStep & operator/=( double x ) {
    m_length           /= x;
    m_radiationLengths /= x;
    m_energyLoss       /= x;
    return *this;
  }
};

inline 
MaterialAccountingStep operator+( const MaterialAccountingStep & x, const MaterialAccountingStep & y ) {
  MaterialAccountingStep step(x);
  step += y;
  return step;
}

inline 
MaterialAccountingStep operator-( const MaterialAccountingStep & x, const MaterialAccountingStep & y ) {
  MaterialAccountingStep step(x);
  step -= y;
  return step;
}

inline 
MaterialAccountingStep operator*( const MaterialAccountingStep & x, const MaterialAccountingStep & y ) {
  MaterialAccountingStep step(x);
  step *= y;
  return step;
}

inline 
MaterialAccountingStep operator*( const MaterialAccountingStep & x, double y ) {
  MaterialAccountingStep step(x);
  step *= y;
  return step;
}

inline 
MaterialAccountingStep operator*( double y, const MaterialAccountingStep & x ) {
  MaterialAccountingStep step(x);
  step *= y;
  return step;
}

inline 
MaterialAccountingStep operator/( const MaterialAccountingStep & x, double y ) {
  MaterialAccountingStep step(x);
  step /= y;
  return step;
}

#endif // MaterialAccountingStep_h
