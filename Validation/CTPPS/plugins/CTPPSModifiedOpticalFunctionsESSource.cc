// authors: Jan Kaspar (jan.kaspar@gmail.com)

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/CTPPSInterpolatedOpticsRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSetCollection.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

class CTPPSModifiedOpticalFunctionsESSource : public edm::ESProducer {
public:
  CTPPSModifiedOpticalFunctionsESSource(const edm::ParameterSet &);
  ~CTPPSModifiedOpticalFunctionsESSource() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> produce(const CTPPSInterpolatedOpticsRcd &);

private:
  edm::ESGetToken<LHCInterpolatedOpticalFunctionsSetCollection, CTPPSInterpolatedOpticsRcd> inputOpticsToken_;

  std::string scenario_;

  double factor_;

  unsigned int rpDecId_45_N_, rpDecId_45_F_, rpDecId_56_N_, rpDecId_56_F_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSModifiedOpticalFunctionsESSource::CTPPSModifiedOpticalFunctionsESSource(const edm::ParameterSet &iConfig)
    : scenario_(iConfig.getParameter<std::string>("scenario")),
      factor_(iConfig.getParameter<double>("factor")),
      rpDecId_45_N_(iConfig.getParameter<unsigned int>("rpId_45_N")),
      rpDecId_45_F_(iConfig.getParameter<unsigned int>("rpId_45_F")),
      rpDecId_56_N_(iConfig.getParameter<unsigned int>("rpId_56_N")),
      rpDecId_56_F_(iConfig.getParameter<unsigned int>("rpId_56_F")) {
  setWhatProduced(this, iConfig.getParameter<std::string>("outputOpticsLabel"))
      .setConsumes(inputOpticsToken_, edm::ESInputTag("", iConfig.getParameter<std::string>("inputOpticsLabel")));
}

//----------------------------------------------------------------------------------------------------

void CTPPSModifiedOpticalFunctionsESSource::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("inputOpticsLabel", "")->setComment("label of the input optics records");
  desc.add<std::string>("outputOpticsLabel", "modified")->setComment("label of the output optics records");

  desc.add<std::string>("scenario", "none")->setComment("name of modification scenario");

  desc.add<double>("factor", 0.)->setComment("size of modification (number of sigmas)");

  desc.add<unsigned int>("rpId_45_N", 0)->setComment("decimal RP id for 45 near");
  desc.add<unsigned int>("rpId_45_F", 0)->setComment("decimal RP id for 45 far");
  desc.add<unsigned int>("rpId_56_N", 0)->setComment("decimal RP id for 56 near");
  desc.add<unsigned int>("rpId_56_F", 0)->setComment("decimal RP id for 56 far");

  descriptions.add("ctppsModifiedOpticalFunctionsESSource", desc);
}

//----------------------------------------------------------------------------------------------------

std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> CTPPSModifiedOpticalFunctionsESSource::produce(
    const CTPPSInterpolatedOpticsRcd &iRecord) {
  // get input
  LHCInterpolatedOpticalFunctionsSetCollection const &input = iRecord.get(inputOpticsToken_);

  // prepare output
  std::shared_ptr<LHCInterpolatedOpticalFunctionsSetCollection> output =
      std::make_shared<LHCInterpolatedOpticalFunctionsSetCollection>(input);

  // premare arm/RP id map
  struct ArmInfo {
    unsigned int rpId_N = 0, rpId_F = 0;
  };

  std::map<unsigned int, ArmInfo> armInfo;

  for (const auto &fsp : *output) {
    CTPPSDetId rpId(fsp.first);
    unsigned int rpDecId = 100 * rpId.arm() + 10 * rpId.station() + rpId.rp();

    if (rpDecId == rpDecId_45_N_)
      armInfo[0].rpId_N = fsp.first;
    if (rpDecId == rpDecId_45_F_)
      armInfo[0].rpId_F = fsp.first;
    if (rpDecId == rpDecId_56_N_)
      armInfo[1].rpId_N = fsp.first;
    if (rpDecId == rpDecId_56_F_)
      armInfo[1].rpId_F = fsp.first;
  }

  // loop over arms
  bool applied = false;

  for (const auto &ap : armInfo) {
    const auto &arm = ap.first;

    //printf("* arm %u\n", arm);

    auto &of_N = output->find(ap.second.rpId_N)->second;
    auto &of_F = output->find(ap.second.rpId_F)->second;

    const double z_N = of_N.getScoringPlaneZ();
    const double z_F = of_F.getScoringPlaneZ();
    const double de_z = (arm == 0) ? z_N - z_F : z_F - z_N;

    //printf("  z_N = %.3f m, z_F = %.3f m\n", z_N*1E-2, z_F*1E-2);

    if (of_N.m_xi_values.size() != of_N.m_xi_values.size())
      throw cms::Exception("CTPPSModifiedOpticalFunctionsESSource")
          << "Different xi sampling of optical functions in near and far RP.";

    // loop over sampling points (in xi)
    for (unsigned int i = 0; i < of_N.m_xi_values.size(); ++i) {
      const double xi = of_N.m_xi_values[i];

      double x_d_N = of_N.m_fcn_values[LHCOpticalFunctionsSet::exd][i];
      double x_d_F = of_F.m_fcn_values[LHCOpticalFunctionsSet::exd][i];

      double L_x_N = of_N.m_fcn_values[LHCOpticalFunctionsSet::eLx][i];
      double L_x_F = of_F.m_fcn_values[LHCOpticalFunctionsSet::eLx][i];
      double Lp_x = of_N.m_fcn_values[LHCOpticalFunctionsSet::eLpx][i];

      //printf("  xi = %.3f, Lp_x = %.3f, %.3f\n", xi, Lp_x, (L_x_F - L_x_N) / de_z);

      // apply modification scenario
      if (scenario_ == "none")
        applied = true;

      if (scenario_ == "Lx") {
        const double a = 3180., b = 40.;  // cm
        const double de_L_x = factor_ * (a * xi + b);
        L_x_N += de_L_x;
        L_x_F += de_L_x;
        applied = true;
      }

      if (scenario_ == "Lpx") {
        const double a = 0.42, b = 0.015;  // dimensionless
        const double de_Lp_x = factor_ * (a * xi + b) * Lp_x;
        Lp_x += de_Lp_x;
        L_x_N -= de_Lp_x * de_z / 2.;
        L_x_F += de_Lp_x * de_z / 2.;
        applied = true;
      }

      if (scenario_ == "xd") {
        const double d = 0.08;  // dimensionless
        x_d_N += x_d_N * d * factor_;
        x_d_F += x_d_F * d * factor_;
        applied = true;
      }

      // TODO: for test only
      if (scenario_ == "Lx-scale") {
        L_x_N *= factor_;
        L_x_F *= factor_;
        applied = true;
      }

      // store updated values
      of_N.m_fcn_values[LHCOpticalFunctionsSet::exd][i] = x_d_N;
      of_F.m_fcn_values[LHCOpticalFunctionsSet::exd][i] = x_d_F;

      of_N.m_fcn_values[LHCOpticalFunctionsSet::eLx][i] = L_x_N;
      of_F.m_fcn_values[LHCOpticalFunctionsSet::eLx][i] = L_x_F;

      of_N.m_fcn_values[LHCOpticalFunctionsSet::eLpx][i] = Lp_x;
      of_F.m_fcn_values[LHCOpticalFunctionsSet::eLpx][i] = Lp_x;
    }

    // re-initialise splines
    of_N.initializeSplines();
    of_F.initializeSplines();
  }

  // modification applied?
  if (!applied)
    edm::LogError("CTPPSModifiedOpticalFunctionsESSource") << "Could not apply scenario `" + scenario_ + "'.";

  // save modified output
  return output;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_MODULE(CTPPSModifiedOpticalFunctionsESSource);
