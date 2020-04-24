#ifndef SimG4CMS_HcalTestBeam_HcalTB06Analysis_H
#define SimG4CMS_HcalTestBeam_HcalTB06Analysis_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Analysis
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
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <memory>
#include <vector>
#include <string>
  
class HcalTB06Histo;

class HcalTB06Analysis : public edm::one::EDAnalyzer<edm::one::SharedResources>
{
public:

  explicit HcalTB06Analysis(const edm::ParameterSet &p);
  ~HcalTB06Analysis() override;

  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event & e, const edm::EventSetup& c) override;

  HcalTB06Analysis(const HcalTB06Analysis&) = delete; 
  const HcalTB06Analysis& operator=(const HcalTB06Analysis&) = delete;

private:

  edm::EDGetTokenT<edm::PCaloHitContainer> m_EcalToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> m_HcalToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> m_BeamToken;
  bool        m_ECAL;

  int         count;
  int         m_idxetaEcal;
  int         m_idxphiEcal;
  int         m_idxetaHcal;
  int         m_idxphiHcal;

  double      m_eta;
  double      m_phi;
  double      m_ener;
  double      m_timeLimit;
  double      m_widthEcal;
  double      m_widthHcal;
  double      m_factEcal;
  double      m_factHcal;
  std::vector<int> m_PDG;

  HcalTB06Histo* m_histo;

};

#endif
