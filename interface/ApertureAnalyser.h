#ifndef ApertureAnalyser_h
#define ApertureAnalyser_h

#include "RPXMLConfig.h"
#include <string>
#include "LHCOpticsApproximator.h"
#include <vector>
#include <fstream>
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"


struct ApertureAnalysisConf;

class ApertureHistogramsEntity
{
  public:
    ApertureHistogramsEntity(std::string aperture_name, ApertureAnalysisConf &conf);
    void AddNamePrefix(std::string prefix);
    void Write();
    void FillApertureHitPositionSingleAperture(MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot, ApertureAnalysisConf &conf);
    void FillApertureHitInfoSingleAperture(MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf);
    void FillApertureHitPositionSelectedSequence(MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot, ApertureAnalysisConf &conf);
    void FillApertureHitInfoSelectedSequence(MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf);
    void FillReferenceHists(MadKinematicDescriptor &in_prot, ApertureAnalysisConf &conf);
    void FillSingleApertureSurvived(MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot, bool out_pos_valid, ApertureAnalysisConf &conf);
    void FillTrackSurvivedSelectedApertures(MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot, bool out_pos_valid, ApertureAnalysisConf &conf);
    void FillTrackLostInSelectedApertures(MadKinematicDescriptor &in_prot, MadKinematicDescriptor &out_prot, bool out_pos_valid, ApertureAnalysisConf &conf);

  private:
    void ComputeOpticsParameters(ApertureAnalysisConf &conf);
    void PreprocessBeforeWriting();
    void AllocateHistograms(ApertureAnalysisConf &conf);
    
    bool CheckIfPhysicalProton(double beam_energy, double ksi, double t);
    double IPSmearedProtonMomentumTot(double px, double py, double pz) const; //GeV
    double CanonicalAnglesTot(double ThetaX, double ThetaY, double Xi) const; //GeV
    
    TH2F aperture_hits_in_selected_squence;
    TH2F t_xi_absorbed_in_selected_sequence;
    TH2F t_xi_absorbtion_acceptance_in_selected_sequence;
    
    TH2F aperture_hits_in_single_aperture;
    TH2F t_xi_absorbed_in_single_aperture;
    TH2F t_xi_absorbtion_acceptance_in_single_aperture;
    
    TH2F t_xi_oryginal_dist;
    TH2F debug_thx_thy_oryginal_dist;
    TH2F debug_x_y_oryginal_dist;
    TH1F debug_xi_oryginal_dist;
    
    //to be done
    TH2F t_xi_surviving_tracks_after_single_aperture;
    TH2F t_xi_surviving_tracks_after_single_aperture_acceptance;
    TH2F debug_x_y_surviving_tracks_at_target_after_single_aperture;
   
    TH2F t_xi_surviving_tracks_after_selected_sequence;
    TH2F t_xi_surviving_tracks_after_selected_sequence_acceptance;
    TH2F debug_x_y_surviving_tracks_at_target_after_selected_sequence;
   
    TH2F t_xi_tracks_lost_in_selected_sequence;
    TH2F t_xi_tracks_lost_in_selected_sequence_acceptance;
    TH2F debug_x_y_tracks_at_target_lost_in_selected_sequence;    
    
    std::string aperture_name_;
    
    //optics related variables
    double proton_mass_;
    double nominal_beam1_px_smeared_;
    double nominal_beam1_py_smeared_;
    double nominal_beam1_pz_smeared_;
    double beam_energy_;
    double beam_momentum_;
};



struct ApertureAnalysisConf
{
  std::string optics_apperture_parametrisation_file;
  std::string optics_apperture_parametrisation_name;
    
  std::string analysis_output_file;
  std::string analysis_output_hist_file;
    
  double t_min;
  double t_max;
  double xi_min;
  double xi_max;
    
  double ip_beta;
  double ip_norm_emit;
  double nominal_beam_energy;
  
  double x_offset;
  double x_half_crossing_angle;
  double y_offset;
  double y_half_crossing_angle;
  
  double y_dest_rp_pos;
  double x_dest_rp_pos;
  bool invert_x;  //true if x:=-x, not implemented yet
  
  int random_seed;
    
  int lost_smaple_population;
};


struct ApertureHitInfo
{
  MadKinematicDescriptor out;
  bool lost;
  bool out_pos_valid;
};

struct ApertureTrackInfo
{
  MadKinematicDescriptor in;
  std::vector<ApertureHitInfo> aperture_hits;
  bool usefull_in_aperture_selection;
};

struct ApertureLossesInfo
{
  int id;
  int rank;
  int particles_lost_if_alone;
  int particles_lost_in_selected_sequence;
  
  double fraction_of_particles_lost_if_alone;
  double fraction_of_particles_lost_in_selected_sequence;
  double fraction_of_particles_lost_in_selected_sequence_vs_remaining_particles;
};


class ApertureAnalyser
{
  public:
    ApertureAnalyser(std::string);
    void AnalyseApertures();
    void Sort(double &max, double &min);
  
  private:
    void OpenXMLConfigurationFile(std::string file_name);
    ApertureAnalysisConf GetParamConfiguration(int id);
    bool ReadParameterisation(ApertureAnalysisConf &conf);
    void BuildTestTracks(ApertureAnalysisConf &conf_);
    void AnalyseApertureLosses(ApertureAnalysisConf &conf_);
    void AnalyseAllApertureAcceptances(ApertureAnalysisConf &conf_);
    void FindTheBottleneckApertures(ApertureAnalysisConf &conf_);
    void OpenInfoTextFile(ApertureAnalysisConf &conf_);
    void CloseInfoTextFile(ApertureAnalysisConf &conf_);
    void WriteApertureHistograms(ApertureAnalysisConf &conf_);
    void AnalyseDestinationAcceptance(ApertureAnalysisConf &conf_);
  
    RPXMLConfig xml_parser_;
    std::string conf_file_name_;
    ApertureAnalysisConf conf_;
    LHCOpticsApproximator parameterisation_;
    std::vector<LHCApertureApproximator> apertures_;
    
    std::vector<ApertureTrackInfo> aperture_tracks_;
    std::vector<ApertureTrackInfo> target_tracks_;
    std::fstream out_file_;
    
    std::vector<ApertureHistogramsEntity> aperture_hists_;
    std::vector<ApertureHistogramsEntity> all_aperture_hists_;
};


#endif
