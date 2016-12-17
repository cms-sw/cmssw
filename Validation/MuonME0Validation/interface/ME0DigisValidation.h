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
    bool isMatched(const int, const int, const int, const int, const int, const int);
private:
    
    MonitorElement* me0_strip_dg_xy[2][6];
    MonitorElement* me0_strip_dg_xy_Muon[2][6];
    MonitorElement* me0_strip_dg_zr[2][6];
    MonitorElement* me0_strip_dg_zr_tot[2];
    MonitorElement* me0_strip_dg_zr_tot_Muon[2];
    
    MonitorElement* me0_strip_dg_dx_local_Muon[2][6];
    MonitorElement* me0_strip_dg_dy_local_Muon[2][6];
    MonitorElement* me0_strip_dg_dphi_global_Muon[2][6];
    
    MonitorElement* me0_strip_dg_dx_local_tot_Muon;
    MonitorElement* me0_strip_dg_dy_local_tot_Muon;
    MonitorElement* me0_strip_dg_x_local_tot;
    MonitorElement* me0_strip_dg_y_local_tot;
    MonitorElement* me0_strip_dg_dphi_global_tot_Muon;
    MonitorElement* me0_strip_dg_dphi_vs_phi_global_tot_Muon;
    MonitorElement* me0_strip_dg_dtime_tot_Muon;
    MonitorElement* me0_strip_dg_time_tot;
    
    MonitorElement* me0_strip_dg_den_eta[2][6];
    MonitorElement* me0_strip_dg_num_eta[2][6];
    
    MonitorElement* me0_strip_dg_den_eta_tot;
    MonitorElement* me0_strip_dg_num_eta_tot;
    
    MonitorElement* me0_strip_dg_bkg_eta_tot;
    MonitorElement* me0_strip_dg_bkgElePos_eta;
    MonitorElement* me0_strip_dg_bkgNeutral_eta;
    
    edm::EDGetToken InputTagToken_Digi;
    
    Int_t npart;
    
};

#endif
