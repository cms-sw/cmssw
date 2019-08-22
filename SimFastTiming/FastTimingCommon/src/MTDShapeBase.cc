#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimFastTiming/FastTimingCommon/interface/MTDShapeBase.h"

MTDShapeBase::~MTDShapeBase() {}

MTDShapeBase::MTDShapeBase()
    : qNSecPerBin_(1. / kNBinsPerNSec), indexOfMax_(0), timeOfMax_(0.), shape_(DVec(k1NSecBinsTotal, 0.0)) {}

std::array<float, 3> MTDShapeBase::timeAtThr(const float scale, const float threshold1, const float threshold2) const {
  std::array<float, 3> times_tmp = {{0., 0., 0.}};

  // --- Check if the pulse amplitude is greater than threshold 2
  if (shape_[indexOfMax_] * scale < threshold2)
    return times_tmp;

  // --- Find the times corresponding to thresholds 1 and 2 on the pulse leading edge
  //     NB: To speed up the search we loop only on the rising edge
  unsigned int index_LT1 = 0;
  unsigned int index_LT2 = 0;

  for (unsigned int i = 0; i < indexOfMax_; ++i) {
    float amplitude = shape_[i] * scale;

    if (amplitude > threshold1 && index_LT1 == 0)
      index_LT1 = i;

    if (amplitude > threshold2 && index_LT2 == 0) {
      index_LT2 = i;
      break;
    }
  }

  // --- Find the time corresponding to thresholds 1 on the pulse falling edge
  unsigned int index_FT1 = 0;

  for (unsigned int i = shape_.size() - 1; i > indexOfMax_; i--) {
    float amplitude = shape_[i] * scale;

    if (amplitude > threshold1 && index_FT1 == 0) {
      index_FT1 = i + 1;
      break;
    }
  }

  if (index_LT1 != 0)
    times_tmp[0] = linear_interpolation(threshold1,
                                        (index_LT1 - 1) * qNSecPerBin_,
                                        index_LT1 * qNSecPerBin_,
                                        shape_[index_LT1 - 1] * scale,
                                        shape_[index_LT1] * scale);

  if (index_LT2 != 0)
    times_tmp[1] = linear_interpolation(threshold2,
                                        (index_LT2 - 1) * qNSecPerBin_,
                                        index_LT2 * qNSecPerBin_,
                                        shape_[index_LT2 - 1] * scale,
                                        shape_[index_LT2] * scale);

  if (index_FT1 != 0)
    times_tmp[2] = linear_interpolation(threshold1,
                                        (index_FT1 - 1) * qNSecPerBin_,
                                        index_FT1 * qNSecPerBin_,
                                        shape_[index_FT1 - 1] * scale,
                                        shape_[index_FT1] * scale);

  return times_tmp;
}

unsigned int MTDShapeBase::indexOfMax() const { return indexOfMax_; }

double MTDShapeBase::timeOfMax() const { return timeOfMax_; }

void MTDShapeBase::buildMe() {
  // --- Fill the vector with the pulse shape
  fillShape(shape_);

  // --- Find the index of maximum
  for (unsigned int i = 0; i < shape_.size(); ++i) {
    if (shape_[indexOfMax_] < shape_[i])
      indexOfMax_ = i;
  }

  if (indexOfMax_ != 0)
    timeOfMax_ = indexOfMax_ * qNSecPerBin_;
}

unsigned int MTDShapeBase::timeIndex(double aTime) const {
  const unsigned int index = aTime * kNBinsPerNSec;

  const bool bad = (k1NSecBinsTotal <= index);

  if (bad)
    LogDebug("MTDShapeBase") << " MTD pulse shape requested for out of range time " << aTime;

  return (bad ? k1NSecBinsTotal : index);
}

double MTDShapeBase::operator()(double aTime) const {
  // return pulse amplitude for request time in ns
  const unsigned int index(timeIndex(aTime));
  return (k1NSecBinsTotal == index ? 0 : shape_[index]);
}

double MTDShapeBase::linear_interpolation(
    const double& y, const double& x1, const double& x2, const double& y1, const double& y2) const {
  if (x1 == x2)
    throw cms::Exception("BadValue") << " MTDShapeBase: Trying to interpolate two points with the same x coordinate!";

  double a = (y2 - y1) / (x2 - x1);
  double b = y1 - a * x1;

  return (y - b) / a;
}
