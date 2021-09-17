#ifndef SimPPS_PPSPixelDigiProducer_RPix_SignalPoint_H
#define SimPPS_PPSPixelDigiProducer_RPix_SignalPoint_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

/**
 * An elementar charge point, with position, sigma from diffusion and tof.
 */
class RPixSignalPoint {
public:
  RPixSignalPoint() : pos_(0, 0), sigma_(0), charge_(0){};

  RPixSignalPoint(double x, double y, double s, double charge) : pos_(x, y), sigma_(s), charge_(charge){};

  inline const LocalPoint& Position() const { return pos_; }
  inline double Sigma() const { return sigma_; }
  inline double Charge() const { return charge_; }

  inline void setCharge(double charge) { charge_ = charge; }
  inline void setPosition(LocalPoint p) { pos_ = p; }
  inline void setSigma(double s) { sigma_ = s; }

private:
  LocalPoint pos_;
  double sigma_;
  double charge_;
};

#endif
