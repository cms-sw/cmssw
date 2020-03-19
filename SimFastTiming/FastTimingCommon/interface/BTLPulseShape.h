#ifndef __SimFastTiming_FastTimingCommon_BTLPulseShape_h__
#define __SimFastTiming_FastTimingCommon_BTLPulseShape_h__

#include "SimFastTiming/FastTimingCommon/interface/MTDShapeBase.h"

class BTLPulseShape : public MTDShapeBase {
public:
  BTLPulseShape();

  ~BTLPulseShape() override;

protected:
  void fillShape(MTDShapeBase::DVec& aVec) const override;
};

#endif
