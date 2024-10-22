#ifndef BasicVertexState_H
#define BasicVertexState_H

#include "TrackingTools/TrajectoryState/interface/ProxyBase11.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalWeight.h"

#include <vector>

class VertexState;

/** Class containing a measurement of a vertex.
 */

class BasicVertexState {
public:
  using Proxy = ProxyBase11<BasicVertexState>;
  using pointer = Proxy::pointer;

public:
  virtual ~BasicVertexState() {}

  template <typename T, typename... Args>
  static std::shared_ptr<BasicVertexState> build(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }

  virtual pointer clone() const = 0;

  /** Access methods
   */
  virtual GlobalPoint position() const = 0;
  virtual GlobalError error() const = 0;
  virtual GlobalError error4D() const = 0;
  virtual double time() const = 0;
  virtual double timeError() const = 0;
  virtual GlobalWeight weight() const = 0;
  virtual GlobalWeight weight4D() const = 0;
  virtual AlgebraicVector3 weightTimesPosition() const = 0;
  virtual AlgebraicVector4 weightTimesPosition4D() const = 0;
  virtual double weightInMixture() const = 0;
  virtual std::vector<VertexState> components() const;
  virtual bool isValid() const = 0;
  virtual bool is4D() const = 0;
};

#endif
