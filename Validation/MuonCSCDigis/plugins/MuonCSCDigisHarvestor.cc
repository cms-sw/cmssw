#ifndef Validation_MuonCSCDigis_MuonCSCDigisHarvestor_h
#define Validation_MuonCSCDigis_MuonCSCDigisHarvestor_h

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Validation/MuonGEMHits/interface/MuonGEMBaseHarvestor.h"

class MuonCSCDigisHarvestor : public MuonGEMBaseHarvestor {
public:
  /// constructor
  explicit MuonCSCDigisHarvestor(const edm::ParameterSet&);
  /// destructor
  ~MuonCSCDigisHarvestor() override {}

  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
};

MuonCSCDigisHarvestor::MuonCSCDigisHarvestor(const edm::ParameterSet& pset)
    : MuonGEMBaseHarvestor(pset, "MuonGEMDigisHarvestor") {}

void MuonCSCDigisHarvestor::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  std::string eff_folder = "MuonCSCDigisV/CSCDigiTask/Stub/Efficiency/";
  std::string occ_folder = "MuonCSCDigisV/CSCDigiTask/Stub/Occupancy/";

  for (int i = 1; i <= 10; ++i) {
    const std::string cn(CSCDetId::chamberName(i));
    std::string d1 = occ_folder + "ALCTEtaDenom_" + cn;
    std::string d2 = occ_folder + "CLCTEtaDenom_" + cn;
    std::string d3 = occ_folder + "LCTEtaDenom_" + cn;

    std::string n1 = occ_folder + "ALCTEtaNum_" + cn;
    std::string n2 = occ_folder + "CLCTEtaNum_" + cn;
    std::string n3 = occ_folder + "LCTEtaNum_" + cn;

    std::string e1 = "ALCTEtaEff_" + cn;
    std::string e2 = "CLCTEtaEff_" + cn;
    std::string e3 = "LCTEtaEff_" + cn;

    bookEff1D(booker, getter, n1, d1, eff_folder, e1, cn + " ALCT Efficiency;True Muon |#eta|;Efficiency");
    bookEff1D(booker, getter, n2, d2, eff_folder, e2, cn + " CLCT Efficiency;True Muon |#eta|;Efficiency");
    bookEff1D(booker, getter, n3, d3, eff_folder, e3, cn + " LCT Efficiency;True Muon |#eta|;Efficiency");
  }
}

DEFINE_FWK_MODULE(MuonCSCDigisHarvestor);
#endif
