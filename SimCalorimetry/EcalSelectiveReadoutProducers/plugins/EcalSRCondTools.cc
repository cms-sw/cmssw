//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 *
 * author: Ph Gras. June, 2010
 */

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/namespace_ecalsrcondtools.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>
#include <iostream>
#include <algorithm>

namespace module {
  class EcalSRCondTools : public edm::one::EDAnalyzer<> {
    //methods
  public:
    /** Constructor
     * @param ps analyser configuration
   */
    EcalSRCondTools(const edm::ParameterSet&);

    /** Destructor
     */
    ~EcalSRCondTools() override;

    /** Called by CMSSW event loop
     * @param evt the event
     * @param es events setup
     */
    void analyze(const edm::Event& evt, const edm::EventSetup& es) override;

    //fields
  private:
    const edm::ParameterSet ps_;

    const std::string mode_;
    bool iomode_write_;
    bool done_;

    edm::ESGetToken<EcalSRSettings, EcalSRSettingsRcd> hSrToken_;
    edm::ESGetToken<EcalTPGPhysicsConst, EcalTPGPhysicsConstRcd> tpgPhysicsConstToken_;
  };

  using namespace std;
  using namespace ecalsrcondtools;

  EcalSRCondTools::EcalSRCondTools(const edm::ParameterSet& ps)
      : ps_(ps), mode_(ps.getParameter<string>("mode")), iomode_write_(true), done_(false) {
    if (mode_ == "read") {
      iomode_write_ = false;
      hSrToken_ = esConsumes();
      tpgPhysicsConstToken_ = esConsumes();
    }
  }

  EcalSRCondTools::~EcalSRCondTools() {}

  void EcalSRCondTools::analyze(const edm::Event& event, const edm::EventSetup& es) {
    if (done_)
      return;
    EcalSRSettings sr;

    if (mode_ == "online_config" || mode_ == "combine_config") {
      string fname = ps_.getParameter<string>("onlineSrpConfigFile");
      ifstream f(fname.c_str());
      if (!f.good()) {
        throw cms::Exception("EcalSRCondTools") << "Failed to open file " << fname;
      }
      importSrpConfigFile(sr, f, true);
    }

    if (mode_ == "python_config" || mode_ == "combine_config") {
      importParameterSet(sr, ps_);
    }

    if (!(mode_ == "python_config" || mode_ == "online_config" || mode_ == "combine_config" || (mode_ == "read"))) {
      throw cms::Exception("Config") << "Invalid value," << mode_ << ",  for parameter mode. "
                                     << "Valid values: online_config, python_config, combine_config, read";
    }

    if (iomode_write_) {
      sr.bxGlobalOffset_ = ps_.getParameter<int>("bxGlobalOffset");
      sr.automaticSrpSelect_ = ps_.getParameter<int>("automaticSrpSelect");
      sr.automaticMasks_ = ps_.getParameter<int>("automaticMasks");

      edm::Service<cond::service::PoolDBOutputService> db;
      if (!db.isAvailable()) {
        throw cms::Exception("CondDBAccess") << "Failed to connect to PoolDBOutputService\n";
      }
      //fillup DB
      //create new infinite IOV
      cond::Time_t firstSinceTime = db->beginOfTime();
      db->writeOneIOV(sr, firstSinceTime, "EcalSRSettingsRcd");
      done_ = true;
    } else {  //read mode
      const edm::ESHandle<EcalSRSettings> hSr = es.getHandle(hSrToken_);
      if (!hSr.isValid()) {
        cout << "EcalSRSettings record not found. Check the Cond DB Global tag.\n";
      } else {
        const EcalSRSettings* ssr = hSr.product();
        cout << "ECAL Seletive readout settings:\n";
        cout << *ssr << "\n" << endl;
      }

      //trigger tower thresholds (from FENIX configuration):
      const edm::ESHandle<EcalTPGPhysicsConst> hTp = es.getHandle(tpgPhysicsConstToken_);
      if (!hTp.isValid()) {
        cout << "EcalTPGPhysicsConst record not found. Check the Cond DB Global tag.\n";
      } else {
        const EcalTPGPhysicsConst* tp = hTp.product();
        const EcalTPGPhysicsConstMap& mymap = tp->getMap();
        if (mymap.size() != 2) {
          cout << "Error: TPG physics record is of unexpected size: " << mymap.size()
               << " elements instead of two (one for EB, one for EE)\n";
        } else {
          EcalTPGPhysicsConstMap::const_iterator it = mymap.begin();
          cout << "----------------------------------------------------------------------\n"
                  "Trigger tower Et thresholds extracted from TPG configuration \n"
                  "(EcalSRCondTools modules supports only read mode for these parameters):\n\n";
          cout << "EB: "
               << "LT = " << it->second.ttf_threshold_Low << " GeV "
               << "HT = " << it->second.ttf_threshold_High << " GeV\n";
          ++it;
          cout << "EE: "
               << "LT = " << it->second.ttf_threshold_Low << " GeV "
               << "HT = " << it->second.ttf_threshold_High << " GeV\n";
        }
      }
    }
  }
}  // namespace module

using namespace module;
DEFINE_FWK_MODULE(EcalSRCondTools);
