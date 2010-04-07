#ifndef BasicVertexState_H
#define BasicVertexState_H

#include "TrackingTools/TrajectoryState/interface/ProxyBase.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "TrackingTools/TrajectoryState/interface/CopyUsingClone.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalWeight.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
//#include "CommonReco/CommonVertex/interface/RefCountedVertexSeed.h"

#include <vector>

class VertexState;

/** Class containing a measurement of a vertex.
 */

class BasicVertexState  : public ReferenceCounted {

public:

  typedef ProxyBase< BasicVertexState, CopyUsingClone<BasicVertexState> > Proxy;
  typedef ReferenceCountingPointer<BasicVertexState>    		  RCPtr;

private:
  //
  // HELP !  new G++ refuses friend class Proxy;
  //
  friend class   ProxyBase< BasicVertexState, CopyUsingClone<BasicVertexState> >;
  friend  class ReferenceCountingPointer<BasicVertexState>    		  ;

public:

  virtual ~BasicVertexState() {}

  virtual BasicVertexState* clone() const = 0;

  /** Access methods
   */
  virtual GlobalPoint position() const = 0;
  virtual GlobalError error() const = 0;
  virtual GlobalWeight weight() const = 0;
  virtual AlgebraicVector3 weightTimesPosition() const = 0;
  virtual double weightInMixture() const = 0;
  virtual std::vector<VertexState> components() const;
  virtual bool isValid() const = 0;


  /** conversion to VertexSeed
   */
//   virtual RefCountedVertexSeed seedWithoutTracks() const = 0;
};

#endif
