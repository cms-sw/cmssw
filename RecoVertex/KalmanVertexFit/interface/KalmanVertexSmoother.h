#ifndef KalmanVertexSmoother_H
#define KalmanVertexSmoother_H

#include "RecoVertex/VertexTools/interface/SequentialVertexSmoother.h"

class KalmanVertexSmoother : public SequentialVertexSmoother<5> {
  /**
   *  A standard vertex smoother: the SequentialVertexSmoother
   *  with standard Kalman tools.
   */
public:
  KalmanVertexSmoother();
};

#endif
