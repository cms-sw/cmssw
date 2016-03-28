#ifndef ME0DigisValidation_H
#define ME0DigisValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"
//Data Formats
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"


class ME0DigisValidation : public ME0BaseValidation
{
public:
  explicit ME0DigisValidation( const edm::ParameterSet& );
  ~ME0DigisValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  MonitorElement* me0_strip_dg_xy[2][6];
  MonitorElement* me0_strip_dg_xy_Muon[2][6];
  MonitorElement* me0_strip_dg_zr[2][6];
  MonitorElement* me0_strip_dg_zr_tot[2];
  MonitorElement* me0_strip_dg_zr_tot_Muon[2];

  edm::EDGetToken InputTagToken_Digi;

  Int_t npart;


};

#endif
