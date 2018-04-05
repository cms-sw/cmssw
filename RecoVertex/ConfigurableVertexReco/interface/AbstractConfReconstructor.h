#ifndef _AbstractConfReconstructor_H_
#define _AbstractConfReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 *  An abstract configurable reconstructor.
 *  must be configurable via ::configure
 */

class AbstractConfReconstructor : public VertexReconstructor
{
  public:

    /** The configure method configures the vertex reconstructor.
     *  It also should also write all its applied defaults back into the map,
     */
    virtual void configure ( const edm::ParameterSet & ) = 0;
    virtual edm::ParameterSet defaults() const = 0;
    ~AbstractConfReconstructor() override {};
    AbstractConfReconstructor * clone() const override = 0;
};

#endif
