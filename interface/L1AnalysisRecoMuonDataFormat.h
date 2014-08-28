#ifndef __L1Analysis_L1AnalysisRecoMuonDataFormat_H__
#define __L1Analysis_L1AnalysisRecoMuonDataFormat_H__

//-------------------------------------------------------------------------------
// Created 20/04/2010 - E. Conte, A.C. Le Bihan
// 
//
// Original code : L1TriggerDPG/L1Ntuples/L1RecoMuonDataFormatProducer - Luigi Guiducci
//-------------------------------------------------------------------------------

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisRecoMuonDataFormat
  {
    L1AnalysisRecoMuonDataFormat(){Reset();};
    ~L1AnalysisRecoMuonDataFormat(){Reset();};
    
    void Reset()
    {
    nMuons = 0;

    // what muon kind? 0=global 1=SA 2=trackeronly
    type.clear();
    howmanytypes.clear();
    // global muons quantities
    ch.clear();  
    pt.clear();
    p.clear(); 
    eta.clear();  
    phi.clear();
    validhits.clear();
    numberOfMatchedStations.clear();
    numberOfValidMuonHits.clear();
    normchi2.clear();
    imp_point_x.clear(); 
    imp_point_y.clear(); 
    imp_point_z.clear(); 
    imp_point_p.clear(); 
    imp_point_pt.clear(); 
    phi_hb.clear();
    z_hb.clear();
    r_he_p.clear();
    r_he_n.clear();
    phi_he_p.clear();
    phi_he_n.clear();

    // tracker muons quantities
    tr_ch.clear();  
    tr_pt.clear();
    tr_p.clear(); 
    tr_eta.clear();  
    tr_phi.clear();
    tr_validhits.clear();
    tr_validpixhits.clear();
    tr_normchi2.clear();
    tr_d0.clear();
    tr_imp_point_x.clear(); 
    tr_imp_point_y.clear(); 
    tr_imp_point_z.clear(); 
    tr_imp_point_p.clear(); 
    tr_imp_point_pt.clear();
 
    tr_z_mb2.clear();
    tr_phi_mb2.clear();
    tr_r_me2_p.clear();
    tr_phi_me2_p.clear();
    tr_r_me2_n.clear();
    tr_phi_me2_n.clear();

    tr_z_mb1.clear();
    tr_phi_mb1.clear();
    tr_r_me1_p.clear();
    tr_phi_me1_p.clear();
    tr_r_me1_n.clear();
    tr_phi_me1_n.clear();

    // standalone muons (either part of global or SA only)
    sa_phi_mb2.clear();
    sa_z_mb2.clear();
    sa_pseta.clear();
    sa_normchi2.clear();
    sa_validhits.clear();
    sa_ch.clear(); 
    sa_pt.clear(); 
    sa_p.clear(); 
    sa_eta.clear(); 
    sa_phi.clear(); 
    sa_outer_pt.clear(); 
    sa_inner_pt.clear(); 
    sa_outer_eta.clear(); 
    sa_inner_eta.clear(); 
    sa_outer_phi.clear(); 
    sa_inner_phi.clear(); 
    sa_outer_x.clear(); 
    sa_outer_y.clear(); 
    sa_outer_z.clear(); 
    sa_inner_x.clear(); 
    sa_inner_y.clear(); 
    sa_inner_z.clear(); 
    sa_imp_point_x.clear(); 
    sa_imp_point_y.clear(); 
    sa_imp_point_z.clear(); 
    sa_imp_point_p.clear(); 
    sa_imp_point_pt.clear(); 
    sa_phi_hb.clear();
    sa_z_hb.clear();
    sa_r_he_p.clear();
    sa_r_he_n.clear();
    sa_phi_he_p.clear();
    sa_phi_he_n.clear();
    sa_phi_me2_p.clear();
    sa_phi_me2_n.clear();
    sa_r_me2_p.clear();
    sa_r_me2_n.clear();

    sa_z_mb1.clear();
    sa_phi_mb1.clear();
    sa_r_me1_p.clear();
    sa_phi_me1_p.clear();
    sa_r_me1_n.clear();
    sa_phi_me1_n.clear();
 
    calo_energy.clear();
    calo_energy3x3.clear();
    ecal_time.clear();
    ecal_terr.clear();
    hcal_time.clear();
    hcal_terr.clear();
    time_dir.clear(); // -1 = outsideIn 0=undefined 1=insideOut
    time_inout.clear();
    time_inout_err.clear();
    time_outin.clear();
    time_outin_err.clear();

    sa_nChambers.clear();
    sa_nMatches.clear();

    // RECHIT information from CSC: only for standalone/global muons!
    rchCSCtype.clear();
    rchPhi.clear();
    rchEta.clear();

    // Trigger matching information:
    hlt_isomu.clear();
    hlt_mu.clear();
    hlt_isoDeltaR.clear();
    hlt_deltaR.clear();


    }      
    
    // how many muons of any kind
    int nMuons;
 
    // what muon kind? 0=global 1=SA 2=trackeronly 3=trsa
    std::vector<int> type;
    std::vector<int> howmanytypes;
    // global muons quantities
    std::vector<double>	ch;  
    std::vector<double>	pt;
    std::vector<double>	p; 
    std::vector<double>	eta;  
    std::vector<double>	phi;
    std::vector<double>	validhits;
    std::vector<double>	numberOfMatchedStations;
    std::vector<double>	numberOfValidMuonHits;
    std::vector<double>	normchi2;
    std::vector<double> imp_point_x; 
    std::vector<double> imp_point_y; 
    std::vector<double> imp_point_z; 
    std::vector<double> imp_point_p; 
    std::vector<double> imp_point_pt; 
    std::vector<double>	phi_hb;
    std::vector<double>	z_hb;
    std::vector<double>	r_he_p;
    std::vector<double>	r_he_n;
    std::vector<double>	phi_he_p;
    std::vector<double>	phi_he_n;

    // tracker muons quantities
    std::vector<double>	tr_ch;  
    std::vector<double>	tr_pt;
    std::vector<double>	tr_p; 
    std::vector<double>	tr_eta;  
    std::vector<double>	tr_phi;
    std::vector<double>	tr_validhits;
    std::vector<double>	tr_validpixhits;    
    std::vector<double>	tr_normchi2;
    std::vector<double>	tr_d0;
    std::vector<double> tr_imp_point_x; 
    std::vector<double> tr_imp_point_y; 
    std::vector<double> tr_imp_point_z; 
    std::vector<double> tr_imp_point_p; 
    std::vector<double> tr_imp_point_pt; 

    std::vector<double> tr_z_mb2;
    std::vector<double> tr_phi_mb2;
    std::vector<double> tr_r_me2_p;
    std::vector<double> tr_phi_me2_p;
    std::vector<double> tr_r_me2_n;
    std::vector<double> tr_phi_me2_n;

    std::vector<double> tr_z_mb1 ;
    std::vector<double> tr_phi_mb1; 
    std::vector<double> tr_r_me1_p ;
    std::vector<double> tr_phi_me1_p; 
    std::vector<double> tr_r_me1_n ;
    std::vector<double> tr_phi_me1_n; 

    // standalone muons (either part of global or SA only)
    std::vector<double>	sa_phi_mb2;
    std::vector<double>	sa_z_mb2;
    std::vector<double>	sa_pseta;
    std::vector<double> sa_normchi2;
    std::vector<double> sa_validhits;
    std::vector<double>	sa_ch; 
    std::vector<double>	sa_pt; 
    std::vector<double>	sa_p; 
    std::vector<double> sa_eta; 
    std::vector<double> sa_phi; 
    std::vector<double> sa_outer_pt; 
    std::vector<double> sa_inner_pt; 
    std::vector<double> sa_outer_eta; 
    std::vector<double> sa_inner_eta; 
    std::vector<double> sa_outer_phi; 
    std::vector<double> sa_inner_phi; 
    std::vector<double>	sa_outer_x; 
    std::vector<double>	sa_outer_y; 
    std::vector<double>	sa_outer_z; 
    std::vector<double>	sa_inner_x; 
    std::vector<double>	sa_inner_y; 
    std::vector<double>	sa_inner_z; 
    std::vector<double> sa_imp_point_x; 
    std::vector<double> sa_imp_point_y; 
    std::vector<double> sa_imp_point_z; 
    std::vector<double> sa_imp_point_p; 
    std::vector<double> sa_imp_point_pt; 
    std::vector<double>	sa_phi_hb;
    std::vector<double>	sa_z_hb;
    std::vector<double>	sa_r_he_p;
    std::vector<double>	sa_r_he_n;
    std::vector<double> sa_phi_he_p;
    std::vector<double> sa_phi_he_n;
    std::vector<double>	sa_r_me2_p;
    std::vector<double>	sa_r_me2_n;
    std::vector<double> sa_phi_me2_p;
    std::vector<double> sa_phi_me2_n;

    std::vector<double> sa_z_mb1;
    std::vector<double> sa_phi_mb1;
    std::vector<double> sa_r_me1_p;
    std::vector<double> sa_phi_me1_p;
    std::vector<double> sa_r_me1_n;
    std::vector<double> sa_phi_me1_n;

    std::vector<double> calo_energy;
    std::vector<double> calo_energy3x3;
    std::vector<double> ecal_time;
    std::vector<double> ecal_terr;
    std::vector<double> hcal_time;
    std::vector<double> hcal_terr;

    std::vector<double> time_dir; // -1 = outsideIn ; 0=undefined; 1=insideOut
    std::vector<double> time_inout;
    std::vector<double> time_inout_err;
    std::vector<double> time_outin;
    std::vector<double> time_outin_err;

    std::vector<int> sa_nChambers;
    std::vector<int> sa_nMatches;

    
    // RECHIT information from CSC: only for standalone/global muons!
    std::vector<int>    rchCSCtype;
    std::vector<double> rchPhi;
    std::vector<double> rchEta;    

    // Trigger matching information:
    std::vector<int> hlt_isomu;
    std::vector<int> hlt_mu;
    std::vector<double> hlt_isoDeltaR;
    std::vector<double> hlt_deltaR;

  }; 
}
#endif


