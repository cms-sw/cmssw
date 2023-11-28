#ifndef RecoTracker_PixelLowPtUtilities_SlidingPeakFinder_h
#define RecoTracker_PixelLowPtUtilities_SlidingPeakFinder_h

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

class SlidingPeakFinder {
public:
  SlidingPeakFinder(unsigned int size) : size_(size), half_((size + 1) / 2) {}

  template <typename Test>
  bool apply(const uint8_t *x,
             const uint8_t *begin,
             const uint8_t *end,
             const Test &test,
             bool verbose = false,
             int firststrip = 0) {
    const uint8_t *ileft = (x != begin) ? std::min_element(x - 1, x + half_) : begin - 1;
    const uint8_t *iright = ((x + size_) < end) ? std::min_element(x + half_, std::min(x + size_ + 1, end)) : end;
    uint8_t left = (ileft < begin ? 0 : *ileft);
    uint8_t right = (iright >= end ? 0 : *iright);
    uint8_t center = *std::max_element(x, std::min(x + size_, end));
    uint8_t maxmin = std::max(left, right);
    if (maxmin < center) {
      bool ret = test(center, maxmin);
      if (ret) {
        ret = test(ileft, iright, begin, end);
      }
      return ret;
    } else {
      return false;
    }
  }

  template <typename V, typename Test>
  bool apply(const V &ampls, const Test &test, bool verbose = false, int firststrip = 0) {
    const uint8_t *begin = &*ampls.begin();
    const uint8_t *end = &*ampls.end();
    for (const uint8_t *x = begin; x < end - (half_ - 1); ++x) {
      if (apply(x, begin, end, test, verbose, firststrip)) {
        return true;
      }
    }
    return false;
  }

private:
  unsigned int size_, half_;
};

struct PeakFinderTest {
  PeakFinderTest(float mip,
                 uint32_t detid,
                 uint32_t firstStrip,
                 const SiStripNoises *theNoise,
                 float seedCutMIPs,
                 float seedCutSN,
                 float subclusterCutMIPs,
                 float subclusterCutSN)
      : mip_(mip),
        detid_(detid),
        firstStrip_(firstStrip),
        noiseObj_(theNoise),
        noises_(theNoise->getRange(detid)),
        subclusterCutMIPs_(subclusterCutMIPs),
        sumCut_(subclusterCutMIPs_ * mip_),
        subclusterCutSN2_(subclusterCutSN * subclusterCutSN) {
    cut_ = std::min<float>(seedCutMIPs * mip, seedCutSN * noiseObj_->getNoise(firstStrip + 1, noises_));
  }

  bool operator()(uint8_t max, uint8_t min) const { return max - min > cut_; }
  bool operator()(const uint8_t *left, const uint8_t *right, const uint8_t *begin, const uint8_t *end) const {
    int yleft = (left < begin ? 0 : *left);
    int yright = (right >= end ? 0 : *right);
    float sum = 0.0;
    int maxval = 0;
    float noise = 0;
    for (const uint8_t *x = left + 1; x < right; ++x) {
      int baseline = (yleft * int(right - x) + yright * int(x - left)) / int(right - left);
      sum += int(*x) - baseline;
      noise += std::pow(noiseObj_->getNoise(firstStrip_ + int(x - begin), noises_), 2);
      maxval = std::max(maxval, int(*x) - baseline);
    }
    if (sum > sumCut_ && sum * sum > noise * subclusterCutSN2_)
      return true;
    return false;
  }

private:
  float mip_;
  unsigned int detid_;
  int firstStrip_;
  const SiStripNoises *noiseObj_;
  SiStripNoises::Range noises_;
  uint8_t cut_;
  float subclusterCutMIPs_, sumCut_, subclusterCutSN2_;
};
#endif
