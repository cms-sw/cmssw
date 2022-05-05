#ifndef PixelSimHitExtraInfo_h
#define PixelSimHitExtraInfo_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include <vector>
#include <cstdint>

class PixelSimHitExtraInfo {
public:
  PixelSimHitExtraInfo(size_t Hindex, const Local3DPoint& entryP, const Local3DPoint& exitP, unsigned int ch) {
    index_ = Hindex;
    theEntryPoint_ = entryP;
    theExitPoint_ = exitP;
    chan_.push_back(ch);
  };
  PixelSimHitExtraInfo() = default;
  ~PixelSimHitExtraInfo() = default;
  size_t hitIndex() const { return index_; };
  const Local3DPoint& entryPoint() const { return theEntryPoint_; };
  const Local3DPoint& exitPoint() const { return theExitPoint_; }
  const std::vector<unsigned int>& channel() const { return chan_; };

  inline bool operator<(const PixelSimHitExtraInfo& other) const { return hitIndex() < other.hitIndex(); }

  void addDigiInfo(unsigned int theDigiChannel) { chan_.push_back(theDigiChannel); }
  bool isInTheList(unsigned int channelToCheck) {
    bool result_in_the_list = false;
    for (unsigned int icheck = 0; icheck < chan_.size(); icheck++) {
      if (channelToCheck == chan_[icheck]) {
        result_in_the_list = true;
        break;
      }
    }
    return result_in_the_list;
  }

private:
  size_t index_;
  Local3DPoint theEntryPoint_;
  Local3DPoint theExitPoint_;
  std::vector<unsigned int> chan_;
};
#endif
