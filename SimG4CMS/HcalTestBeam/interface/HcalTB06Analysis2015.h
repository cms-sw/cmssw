#ifndef SimG4CMS_HcalTestBeam_HcalTB06Analysis2015_H
#define SimG4CMS_HcalTestBeam_HcalTB06Analysis2015_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Analysis2015
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <vector>
#include <string>
  
class HcalTB06Histo;

class HcalTB06Analysis2015 : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:

  explicit HcalTB06Analysis2015(const edm::ParameterSet &p);
  virtual ~HcalTB06Analysis2015();

  virtual void beginJob() override;
  virtual void endJob() override;
  virtual void analyze(const edm::Event & e, const edm::EventSetup& c) override;

private:

  HcalTB06Analysis2015(const HcalTB06Analysis2015&) = delete; 
  const HcalTB06Analysis2015& operator=(const HcalTB06Analysis2015&) = delete;

  int         count;
  edm::InputTag m_EcalTag;
  edm::InputTag m_HcalTag;
  bool        m_ECAL;

  double      m_eta;
  double      m_phi;
  double      m_ener;
  double      m_widthEcal;
  double      m_factEcal;
  double      m_factHcal;
  std::vector<int> m_PDG;

  HcalTB06Histo* m_histo;

};

#endif
