#ifndef _ReconstructorFromFitter_H_
#define _ReconstructorFromFitter_H_

#include <memory>
#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"

/**
 *  Wrap any VertexFitter into the VertexReconstructor interface
 */

class ReconstructorFromFitter : public AbstractConfReconstructor {
public:
  explicit ReconstructorFromFitter(std::unique_ptr<AbstractConfFitter>&&);
  ReconstructorFromFitter(const ReconstructorFromFitter& o);
  ~ReconstructorFromFitter() override;
  void configure(const edm::ParameterSet&) override;
  edm::ParameterSet defaults() const override;
  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>&) const override;
  std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>&, const reco::BeamSpot&) const override;

  ReconstructorFromFitter* clone() const override;

private:
  const AbstractConfFitter* theFitter;
};

#endif
