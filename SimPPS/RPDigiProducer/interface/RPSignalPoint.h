#ifndef SimPPS_RPDigiProducer_RP_SignalPoint_H
#define SimPPS_RPDigiProducer_RP_SignalPoint_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"


/**
 * An elementar charge point, with position, sigma from diffusion and tof.
 */
class RPSignalPoint {
  public:
    RPSignalPoint() : pos_(0,0), sigma_(0), charge_(0) {}
    
    RPSignalPoint(double x, double y, double s, double charge) : 
      pos_(x,y), sigma_(s), charge_(charge) {}
    
    inline LocalPoint Position() const { return pos_;}
    inline LocalPoint &Position() { return pos_;}
    inline double X() const {return pos_.x();}
    inline double Y() const { return pos_.y();}
    inline double Sigma() const { return sigma_;}
    inline double & Sigma() { return sigma_;}
    inline double Charge() const { return charge_;}
    inline double & Charge() { return charge_;}
    inline void SetCharge(double charge) { charge_ = charge;} 
  private:
    LocalPoint pos_;
    double sigma_;
    double charge_;
};

#endif  //SimPPS_RPDigiProducer_RP_SignalPoint_H
