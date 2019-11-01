// -*- C++ -*-
//
// Package:    HcalTestNumberingTester
// Class:      HcalTestNumberingTester
//
/**\class HcalTestNumberingTester HcalTestNumberingTester.cc test/HcalTestNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2013/12/26
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"

class HcalTestNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalTestNumberingTester(const edm::ParameterSet&);
  ~HcalTestNumberingTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

HcalTestNumberingTester::HcalTestNumberingTester(const edm::ParameterSet&) {}

HcalTestNumberingTester::~HcalTestNumberingTester() {}

// ------------ method called to produce the data  ------------
void HcalTestNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<HcalDDDSimConstants> pHSNDC;
  iSetup.get<HcalSimNumberingRecord>().get(pHSNDC);
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  iSetup.get<HcalRecNumberingRecord>().get(pHRNDC);

  if (pHSNDC.isValid() && pHRNDC.isValid()) {
    std::cout << "about to de-reference the edm's" << std::endl;
    HcalDDDSimConstants* hcs = (HcalDDDSimConstants*)(&(*pHSNDC));
    HcalDDDRecConstants* hcr = (HcalDDDRecConstants*)(&(*pHRNDC));
    HcalNumberingScheme* schme1 = new HcalNumberingScheme();
    HcalNumberingScheme* schme2 = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(false));

    for (int type = 0; type < 2; ++type) {
      HcalSubdetector sub = (type == 0) ? HcalBarrel : HcalEndcap;
      for (int zs = 0; zs < 2; ++zs) {
        int zside = 2 * zs - 1;
        std::pair<int, int> etas = hcr->getEtaRange(type);
        for (int eta = etas.first; eta <= etas.second; ++eta) {
          std::vector<std::pair<int, double> > phis = hcr->getPhis(sub, eta);
          for (unsigned int k = 0; k < phis.size(); ++k) {
            int phi = phis[k].first;
            int lmin = (type == 1 && eta == 16) ? 8 : 1;
            int lmax = (type == 1) ? 19 : ((eta == 16) ? 7 : 17);
            for (int lay = lmin; lay <= lmax; ++lay) {
              std::pair<int, int> etd = hcs->getEtaDepth(sub, eta, phi, zside, 0, lay);
              HcalNumberingFromDDD::HcalID tmp(sub, zs, etd.second, etd.first, phi, phi, lay);
              uint32_t id1 = schme1->getUnitID(tmp);
              uint32_t id2 = schme2->getUnitID(tmp);
              DetId id0 = HcalHitRelabeller::relabel(id2, hcr);
              std::cout << "I/P " << sub << ":" << zside * eta << ":" << phi << ":" << lay << " Normal " << std::hex
                        << id1 << std::dec << " " << HcalDetId(id1) << " Test " << std::hex << id2 << std::dec << " "
                        << HcalDetId(id0);
              if (id1 != id0.rawId())
                std::cout << " *** ERROR ***";
              std::cout << std::endl;
            }
          }
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTestNumberingTester);
