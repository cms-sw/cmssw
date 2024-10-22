#ifndef _ConfigurableVertexReconstructor_H_
#define _ConfigurableVertexReconstructor_H_

#include "RecoVertex/VertexPrimitives/interface/VertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"
#include <string>
#include <map>

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ConfigurableVertexReconstructor : public VertexReconstructor {
public:
  ConfigurableVertexReconstructor(const edm::ParameterSet &);
  ConfigurableVertexReconstructor(const ConfigurableVertexReconstructor &o);
  ~ConfigurableVertexReconstructor() override;

  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &) const override;
  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &,
                                        const reco::BeamSpot &) const override;
  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack> &,
                                        const std::vector<reco::TransientTrack> &,
                                        const reco::BeamSpot &) const override;

  ConfigurableVertexReconstructor *clone() const override;

private:
  AbstractConfReconstructor *theRector;
};

#endif
