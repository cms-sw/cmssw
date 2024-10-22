#ifndef PixelDigiAddTempInfo_h
#define PixelDigiAddTempInfo_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

class PixelDigiAddTempInfo {
public:
  PixelDigiAddTempInfo(unsigned int ch,
                       size_t Hindex,
                       const Local3DPoint& entryP,
                       const Local3DPoint& exitP,
                       int PType,
                       int PartID,
                       uint32_t detID,
                       float InitCharge) {
    chan_ = ch;
    index_ = Hindex;
    theEntryPoint_ = entryP;
    theExitPoint_ = exitP;
    theProcessType_ = PType;
    thePartID_ = PartID;
    detectorID_ = detID;
    charge_ = InitCharge;
  };
  PixelDigiAddTempInfo() = default;
  ~PixelDigiAddTempInfo() = default;
  const unsigned int channel() const { return chan_; };
  const size_t hitIndex() const { return index_; };
  const Local3DPoint& entryPoint() const { return theEntryPoint_; };
  const Local3DPoint& exitPoint() const { return theExitPoint_; }
  const int processType() const { return theProcessType_; };
  const int trackID() const { return thePartID_; };
  const uint32_t detID() const { return detectorID_; };
  const float getCharge() const { return charge_; };
  void addCharge(float charge_to_be_added) { charge_ += charge_to_be_added; };

  inline bool operator<(const PixelDigiAddTempInfo& other) const { return channel() < other.channel(); }

private:
  unsigned int chan_;
  size_t index_;
  Local3DPoint theEntryPoint_;
  Local3DPoint theExitPoint_;
  int theProcessType_;
  int thePartID_;
  uint32_t detectorID_;
  float charge_;
};
#endif
