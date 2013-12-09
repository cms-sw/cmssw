#include "Validation/RecoVertex/interface/VertexWeighter.h"
#include <vector>
#include <math.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"


VertexWeighter::VertexWeighter():
  m_sigma_init(1.), m_mean_init(0.), m_sigma_final(1.), m_usemain(false), m_dummy(true) { }

VertexWeighter::VertexWeighter(const double& sigma_init, const double& mean_init, const double& sigma_final, const bool& usemainvtx):
  m_sigma_init(sigma_init), m_mean_init(mean_init), m_sigma_final(sigma_final), m_usemain(usemainvtx), m_dummy(false) { }

VertexWeighter::VertexWeighter(const edm::ParameterSet& iConfig):
  m_sigma_init(iConfig.getParameter<double>("initSigma")), 
  m_mean_init(iConfig.getParameter<double>("initMean")), 
  m_sigma_final(iConfig.getParameter<double>("finalSigma")), 
  m_usemain(iConfig.getParameter<bool>("useMainVertex")), 
  m_dummy(false) { }

const double VertexWeighter::weight(const std::vector<float>& zpositions, const float& zmain) const {

  double final_weight = 1.;

  if(!m_dummy) {

    for(std::vector<float>::const_iterator zpos = zpositions.begin() ; zpos != zpositions.end() ; ++zpos) {
      
      final_weight *= (m_sigma_init/m_sigma_final) * exp(-pow((*zpos-m_mean_init),2)/2.*(1./pow(m_sigma_final,2)-1./pow(m_sigma_init,2)));

    }

    if(m_usemain) {
      final_weight *= (m_sigma_init/m_sigma_final) * exp(-pow((zmain-m_mean_init),2)/2.*(1./pow(m_sigma_final,2)-1./pow(m_sigma_init,2)));
    }

  }

  return final_weight;


}
