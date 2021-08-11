#ifndef RecoBTag_BasicGhostTrackState_h
#define RecoBTag_BasicGhostTrackState_h

#include <utility>

#include "TrackingTools/TrajectoryState/interface/ProxyBase11.h"

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace reco {

  class GhostTrackPrediction;

  class BasicGhostTrackState {
  public:
    using BGTS = BasicGhostTrackState;
    using Proxy = ProxyBase11<BGTS>;
    using pointer = Proxy::pointer;

    typedef math::Error<3>::type CovarianceMatrix;
    typedef std::pair<GlobalPoint, GlobalError> Vertex;

    virtual ~BasicGhostTrackState() {}

    template <typename T, typename... Args>
    static std::shared_ptr<BGTS> build(Args &&...args) {
      return std::make_shared<T>(std::forward<Args>(args)...);
    }

    virtual GlobalPoint globalPosition() const = 0;
    virtual GlobalError cartesianError() const = 0;
    virtual CovarianceMatrix cartesianCovariance() const = 0;

    double lambda() const { return lambda_; }
    virtual bool isValid() const { return true; }

    virtual void reset() {}
    virtual bool linearize(const GhostTrackPrediction &pred, bool initial, double lambda) {
      lambda_ = lambda;
      return true;
    }
    virtual bool linearize(const GhostTrackPrediction &pred, double lambda) {
      lambda_ = lambda;
      return true;
    }

    virtual Vertex vertexStateOnGhostTrack(const GhostTrackPrediction &pred, bool withMeasurementError) const = 0;
    virtual Vertex vertexStateOnMeasurement(const GhostTrackPrediction &pred, bool withGhostTrackError) const = 0;

    double weight() const { return weight_; }
    void setWeight(double weight) { weight_ = weight; }

    virtual pointer clone() const = 0;

  protected:
    double lambda_ = 0;
    double weight_ = 1.;
  };

}  // namespace reco

#endif  // RecoBTag_BasicGhostTrackState_h
