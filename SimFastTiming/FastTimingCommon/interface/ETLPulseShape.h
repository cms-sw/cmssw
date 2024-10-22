#ifndef __SimFastTiming_FastTimingCommon_ETLPulseShape_h__
#define __SimFastTiming_FastTimingCommon_ETLPulseShape_h__

#include "SimFastTiming/FastTimingCommon/interface/MTDShapeBase.h"

class ETLPulseShape : public MTDShapeBase {
public:
  ETLPulseShape();

  ~ETLPulseShape() override;

protected:
  void fillShape(MTDShapeBase::DVec& aVec) const override;
};

#endif
