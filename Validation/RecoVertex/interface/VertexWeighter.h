#ifndef TRACKINGPFG_PILEUP_VERTEXWEIGHTER_H
#define TRACKINGPFG_PILEUP_VERTEXWEIGHTER_H

#include <vector>

namespace edm { class ParameterSet; }

class VertexWeighter{

 public:

  VertexWeighter();
  VertexWeighter(const double& sigma_init, const double& mean_init, const double& sigma_final, const bool& usemainvtx);
  VertexWeighter(const edm::ParameterSet& iConfig);

  const double weight(const std::vector<float>& zpositions, const float& zmain) const;

 private:

  const double m_sigma_init;
  const double m_mean_init;
  const double m_sigma_final;
  const bool m_usemain;
  const bool m_dummy; 


};


#endif // TRACKINGPFG_PILEUP_VERTEXWEIGHTER_H
