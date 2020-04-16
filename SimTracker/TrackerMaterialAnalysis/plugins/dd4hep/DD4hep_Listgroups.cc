#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>

#include "boost/format.hpp"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <TROOT.h>
#include <TProfile2D.h>
#include <TH2F.h>
#include <TLine.h>
#include <TText.h>
#include <TColor.h>

class DD4hep_ListGroups : public edm::one::EDAnalyzer<> {
public:
  DD4hep_ListGroups(const edm::ParameterSet &iConfig);
  ~DD4hep_ListGroups() override;

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override {}
  void endJob() override;
  const edm::ESInputTag m_tag;

  bool m_saveSummaryPlot;
  std::vector<TH2F *> m_plots;
  std::set<std::string_view> m_group_names;
  std::vector<unsigned int> m_color;
  std::vector<int> m_gradient;
  void fillColor();
  void fillGradient();
  std::vector<std::pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>> overlayEtaReferences();
};

DD4hep_ListGroups::DD4hep_ListGroups(const edm::ParameterSet &iConfig)
    : m_tag(iConfig.getParameter<edm::ESInputTag>("DDDetector")) {
  m_saveSummaryPlot = iConfig.getUntrackedParameter<bool>("SaveSummaryPlot");
}

DD4hep_ListGroups::~DD4hep_ListGroups() {}

void DD4hep_ListGroups::fillColor(void) {
  // With the introduction of the support for PhaseI and PhaseII detectors it
  // became quite difficult to maintain a list of colors that is in sync with
  // the real number of grouping used in the different scenarios. We therefore
  // define some reasonable set and loop over it in case the number of grouping
  // is larger than the number of colors.

  m_color.push_back(kBlack);  // unassigned

  m_color.push_back(kAzure);      // PixelBarrelLayer0_Z0
  m_color.push_back(kAzure - 1);  // PixelBarrelLayer0_Z20
  m_color.push_back(kAzure + 1);  // Layer1_Z0
  m_color.push_back(kAzure + 2);  // Layer1_Z20

  m_color.push_back(kGreen);       // EndCapDisk1_R0
  m_color.push_back(kGreen + 2);   // EndcapDisk1_R11
  m_color.push_back(kGreen + 4);   // EndcapDisk1_R7
  m_color.push_back(kSpring + 9);  // EndcapDisk2_R0
  m_color.push_back(kSpring + 4);  // EndcapDisk2_R7
  m_color.push_back(kSpring);      // EndcapDisk2_R7

  m_color.push_back(kRed);          // TECDisk0_R20
  m_color.push_back(kRed + 2);      // TECDisk0_R40
  m_color.push_back(kRed - 7);      // TECDisk0_R50
  m_color.push_back(kRed - 5);      // TECDisk0_R60
  m_color.push_back(kRed - 10);     // TECDisk0_R90
  m_color.push_back(kRed - 1);      // TECDisk1_Inner
  m_color.push_back(kRed - 2);      // TECDisk1_Outer
  m_color.push_back(kRed - 3);      // TECDisk1_R20
  m_color.push_back(kPink - 2);     // TECDisk2_Inner
  m_color.push_back(kPink - 3);     // TECDisk2_Outer
  m_color.push_back(kPink - 4);     // TECDisk2_R20
  m_color.push_back(kPink + 9);     // TECDisk3_Inner
  m_color.push_back(kPink + 8);     // TECDisk3_Outer
  m_color.push_back(kPink + 7);     // TECDisk3
  m_color.push_back(kMagenta - 2);  // TECDisk4_Inner
  m_color.push_back(kMagenta - 3);  // TECDisk4_Outer
  m_color.push_back(kMagenta - 4);  // TECDisk4_R33
  m_color.push_back(kMagenta - 5);  // TECDisk5_Inner
  m_color.push_back(kMagenta - 6);  // TECDisk5_Outer
  m_color.push_back(kMagenta - 7);  // TECDisk5_R33
  m_color.push_back(kRed);          // TECDisk6
  m_color.push_back(kMagenta - 9);  // TECDisk7_R40
  m_color.push_back(kViolet);       // TECDisk8

  m_color.push_back(kOrange + 9);   // TIBLayer0_Z0
  m_color.push_back(kOrange + 7);   // TIBLayer0_Z20
  m_color.push_back(kOrange + 5);   // TIBLayer0_Z40
  m_color.push_back(kOrange - 2);   // TIBLayer1_Z0
  m_color.push_back(kOrange - 3);   // TIBLayer1_Z30
  m_color.push_back(kOrange - 6);   // TIBLayer1_Z60
  m_color.push_back(kOrange + 4);   // TIBLayer2_Z0
  m_color.push_back(kOrange - 7);   // TIBLayer2_Z40
  m_color.push_back(kOrange);       // TIBLayer3_Z0
  m_color.push_back(kOrange + 10);  // TIBLayer3_Z50

  m_color.push_back(kViolet + 10);  // TIDDisk1_R0
  m_color.push_back(kViolet + 6);   // TIDDisk1_R30
  m_color.push_back(kViolet + 3);   // TIDDisk1_R40
  m_color.push_back(kViolet - 7);   // TIDDisk2_R25
  m_color.push_back(kViolet - 1);   // TIDDisk2_R30
  m_color.push_back(kViolet + 9);   // TIDDisk2_R40
  m_color.push_back(kViolet - 5);   // TIDDisk3_R24
  m_color.push_back(kViolet - 3);   // TIDDisk3_R30
  m_color.push_back(kViolet);       // TIDDisk3_R40

  m_color.push_back(kAzure);      // TOBLayer0_Z0
  m_color.push_back(kAzure + 8);  // TOBLayer0_Z20
  m_color.push_back(kAzure + 2);  // TOBLayer0_Z70
  m_color.push_back(kAzure + 4);  // TOBLayer0_Z80
  m_color.push_back(kCyan + 1);   // TOBLayer1_Z0
  m_color.push_back(kCyan - 9);   // TOBLayer1_Z20
  m_color.push_back(kCyan + 3);   // TOBLayer1_Z80
  m_color.push_back(kCyan + 4);   // TOBLayer1_Z90
  m_color.push_back(kAzure);      // TOBLayer2_Z0
  m_color.push_back(kAzure + 8);  // TOBLayer2_Z25
  m_color.push_back(kAzure + 2);  // TOBLayer2_Z80
  m_color.push_back(kAzure + 5);  // TOBLayer2_Z90
  m_color.push_back(kCyan + 1);   // TOBLayer3_Z0
  m_color.push_back(kCyan - 9);   // TOBLayer3_Z25
  m_color.push_back(kCyan + 3);   // TOBLayer3_Z80
  m_color.push_back(kCyan + 4);   // TOBLayer3_Z90
  m_color.push_back(kAzure);      // TOBLayer4_Z0
  m_color.push_back(kAzure + 8);  // TOBLayer4_Z25
  m_color.push_back(kAzure + 2);  // TOBLayer4_Z80
  m_color.push_back(kAzure + 5);  // TOBLayer4_Z90
  m_color.push_back(kCyan + 1);   // TOBLayer5_Z0
  m_color.push_back(kCyan - 9);   // TOBLayer5_Z25
  m_color.push_back(kCyan + 3);   // TOBLayer5_Z80
  m_color.push_back(kCyan + 4);   // TOBLayer5_Z90
}

void DD4hep_ListGroups::fillGradient() {
  m_gradient.reserve(200);
  unsigned int steps = 100;
  // if no index was given, find the highest used one and start from that plus one
  unsigned int index = ((TObjArray *)gROOT->GetListOfColors())->GetLast() + 1;

  float r1, g1, b1, r2, g2, b2;
  static_cast<TColor *>(gROOT->GetListOfColors()->At(kBlue + 1))->GetRGB(r1, g1, b1);
  static_cast<TColor *>(gROOT->GetListOfColors()->At(kAzure + 10))->GetRGB(r2, g2, b2);
  float delta_r = (r2 - r1) / (steps - 1);
  float delta_g = (g2 - g1) / (steps - 1);
  float delta_b = (b2 - b1) / (steps - 1);

  m_gradient.push_back(kBlue + 4);  // Underflow lowest bin
  unsigned int ii = 0;
  for (unsigned int i = 0; i < steps; ++i, ++ii) {
    new TColor(static_cast<Int_t>(index + ii), r1 + delta_r * i, g1 + delta_g * i, b1 + delta_b * i);
    m_gradient.push_back(index + ii);
  }

  m_gradient.push_back(kWhite);  // 0 level perfectly white

  static_cast<TColor *>(gROOT->GetListOfColors()->At(kOrange))->GetRGB(r1, g1, b1);
  static_cast<TColor *>(gROOT->GetListOfColors()->At(kOrange + 7))->GetRGB(r2, g2, b2);
  delta_r = (r2 - r1) / (steps - 1);
  delta_g = (g2 - g1) / (steps - 1);
  delta_b = (b2 - b1) / (steps - 1);
  for (unsigned int i = 0; i < steps; ++i, ++ii) {
    new TColor(static_cast<Int_t>(index + ii), r1 + delta_r * i, g1 + delta_g * i, b1 + delta_b * i);
    m_gradient.push_back(index + ii);
  }
  m_gradient.push_back(kRed);  // Overflow highest bin
}

std::vector<std::pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>> DD4hep_ListGroups::overlayEtaReferences() {
  std::vector<std::pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>> lines;

  lines.reserve(40);
  std::pair<float, float> deltaZ(293, 298);
  std::pair<float, float> deltaR(115, 118);
  float text_size = 0.033;

  for (float eta = 0.; eta <= 3.8; eta += 0.2) {
    float theta = 2. * atan(exp(-eta));
    if (eta >= 1.8) {
      lines.push_back(std::make_pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>(
          std::make_shared<TLine>(deltaZ.first, deltaZ.first * tan(theta), deltaZ.second, deltaZ.second * tan(theta)),
          std::make_shared<TText>(deltaZ.first, deltaZ.first * tan(theta), str(boost::format("%2.1f") % eta).c_str())));
      lines.back().second->SetTextFont(42);
      lines.back().second->SetTextSize(text_size);
      lines.back().second->SetTextAlign(33);
      lines.push_back(std::make_pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>(
          std::make_shared<TLine>(-deltaZ.first, deltaZ.first * tan(theta), -deltaZ.second, deltaZ.second * tan(theta)),
          std::make_shared<TText>(
              -deltaZ.first, deltaZ.first * tan(theta), str(boost::format("-%2.1f") % eta).c_str())));
      lines.back().second->SetTextFont(42);
      lines.back().second->SetTextSize(text_size);
      lines.back().second->SetTextAlign(13);
    } else {
      lines.push_back(std::make_pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>(
          std::make_shared<TLine>(deltaR.first / tan(theta), deltaR.first, deltaR.second / tan(theta), deltaR.second),
          std::make_shared<TText>(deltaR.first / tan(theta), deltaR.first, str(boost::format("%2.1f") % eta).c_str())));
      lines.back().second->SetTextFont(42);
      lines.back().second->SetTextSize(text_size);
      lines.back().second->SetTextAlign(23);
      if (eta != 0) {
        lines.push_back(std::make_pair<std::shared_ptr<TLine>, std::shared_ptr<TText>>(
            std::make_shared<TLine>(
                -deltaR.first / tan(theta), deltaR.first, -deltaR.second / tan(theta), deltaR.second),
            std::make_shared<TText>(
                -deltaR.first / tan(theta), deltaR.first, str(boost::format("-%2.1f") % eta).c_str())));
        lines.back().second->SetTextFont(42);
        lines.back().second->SetTextSize(text_size);
        lines.back().second->SetTextAlign(23);
      }
    }
  }
  return lines;
}

void DD4hep_ListGroups::analyze(const edm::Event &evt, const edm::EventSetup &setup) {
  edm::ESTransientHandle<cms::DDCompactView> cpv;
  setup.get<IdealGeometryRecord>().get(m_tag, cpv);
  cms::DDFilter filter("TrackingMaterialGroup", "");
  cms::DDFilteredView fv(*cpv, filter);

  for (const auto &t : fv.specpars()) {
    m_group_names.insert(t->strValue("TrackingMaterialGroup"));
    std::cout << t->strValue("TrackingMaterialGroup") << std::endl;
  }

  for (const auto &i : m_group_names) {
    cms::DDFilter filter1("TrackingMaterialGroup", {i.data(), i.size()});
    cms::DDFilteredView fv1(*cpv, filter1);
    bool firstChild = fv1.firstChild();
    //fv1.printFilter();

    std::cout << "***" << i << std::endl;

    for (const auto j : fv1.specpars()) {
      for (const auto k : j->paths) {
        std::cout << k << std::endl;
        if (firstChild) {
          std::cout << "Find children matching selection for a parent " << fv1.name() << "\n";
          std::vector<std::vector<cms::Node *>> children = fv1.children(k);
          for (auto const &path : children) {
            for (auto const &node : path) {
              std::cout << node->GetName() << ", ";
            }
            std::cout << "\n";
          }
          std::cout << "\n";
        }
      }
    }
    std::cout << "*************" << std::endl;
  }
}

void DD4hep_ListGroups::endJob() {}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DD4hep_ListGroups);
