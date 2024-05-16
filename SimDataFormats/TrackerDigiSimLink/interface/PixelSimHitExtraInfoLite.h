#ifndef PixelSimHitExtraInfoLite_h
#define PixelSimHitExtraInfoLite_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include <vector>
#include <cstdint>

class PixelSimHitExtraInfoLite {
public:
  PixelSimHitExtraInfoLite(size_t Hindex, const Local3DPoint& entryP, const Local3DPoint& exitP, unsigned int ch) {
    index_ = Hindex;
//    theEntryPoint_ = entryP;
//    theExitPoint_ = exitP;
    theEntryPointX_ =  MiniFloatConverter::float32to16(entryP.x());
    theEntryPointY_ =  MiniFloatConverter::float32to16(entryP.y());
    theEntryPointZ_ =  MiniFloatConverter::float32to16(entryP.z());
    theExitPointX_ =  MiniFloatConverter::float32to16(exitP.x());
    theExitPointY_ =  MiniFloatConverter::float32to16(exitP.y());
    theExitPointZ_ =  MiniFloatConverter::float32to16(exitP.z());
    chan_.push_back(ch);
  };
  PixelSimHitExtraInfoLite() = default;
  ~PixelSimHitExtraInfoLite() = default;
  size_t hitIndex() const { return index_; };
  const uint16_t& entryPointX()  const { return theEntryPointX_;}
  const uint16_t& entryPointY()  const { return theEntryPointY_;}
  const uint16_t& entryPointZ()  const { return theEntryPointZ_;}
  const uint16_t& exitPointX()  const { return theExitPointX_;}
  const uint16_t& exitPointY()  const { return theExitPointY_;}
  const uint16_t& exitPointZ()  const { return theExitPointZ_;}

  Local3DPoint& entryPoint() { 
	   UncompressedEntryPoint_ = Local3DPoint(
			       MiniFloatConverter::float16to32(theEntryPointX_),
                               MiniFloatConverter::float16to32(theEntryPointY_),
                               MiniFloatConverter::float16to32(theEntryPointZ_) );
	   return UncompressedEntryPoint_;
  }
  Local3DPoint& exitPoint() { 
	   UncompressedExitPoint_ =  Local3DPoint(
			       MiniFloatConverter::float16to32(theExitPointX_),
                               MiniFloatConverter::float16to32(theExitPointY_),
                               MiniFloatConverter::float16to32(theExitPointZ_)  ); 
	   return UncompressedExitPoint_;
  }

  const std::vector<unsigned int>& channel() const { return chan_; };

  inline bool operator<(const PixelSimHitExtraInfoLite& other) const { return hitIndex() < other.hitIndex(); }

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
  uint16_t theEntryPointX_;
  uint16_t theEntryPointY_;
  uint16_t theEntryPointZ_;
  uint16_t theExitPointX_;
  uint16_t theExitPointY_;
  uint16_t theExitPointZ_;
  Local3DPoint UncompressedEntryPoint_;
  Local3DPoint UncompressedExitPoint_;
  std::vector<unsigned int> chan_;
};
#endif
