#include <string>
#include <iostream>
#include <fstream>
#include "TTree.h"
#include "TFile.h"
#include "TMath.h"


#include "TMultiDimFet.h"
#include "LHCOpticsApproximator.h"
#include "RPXMLConfig.h"
#include <boost/shared_ptr.hpp>


struct Parametisation_aperture_configuration;

struct Parametisation_configuration
{
  std::string base_mad_conf_file;
  std::string processed_mad_conf_file;
  std::string base_mad_thin_conf_file;
  std::string beam;
  double nominal_beam_energy;
  std::string from_marker_name;
  double from_marker_s_pos;
  bool define_from;
  std::string to_marker_name;
  double to_marker_s_pos;
  bool define_to;
  bool aperture_limit;
  int tot_entries_number;

  int number_of_part_per_sample;
  double x_min;
  double x_max;
  double theta_x_min;
  double theta_x_max;
  double y_min;
  double y_max;
  double theta_y_min;
  double theta_y_max;
  double ksi_min;
  double ksi_max;

  std::string samples_train_root_file_name;
  std::string samples_test_root_file_name;
  std::string samples_aperture_test_file_name;
  std::string destination_branch_prefix;

  TMultiDimFet::EMDFPolyType polynomials_type;
  LHCOpticsApproximator::polynomials_selection terms_selelection_mode;
  int max_degree_x;
  int max_degree_tx;
  int max_degree_y;
  int max_degree_ty;
  double precision_x;
  double precision_tx;
  double precision_y;
  double precision_ty;

  bool common_terms;
  std::string approximation_error_histogram_file;

  std::string lost_particles_tree_filename;
  std::string lost_particles_hist_filename;

  std::string optics_parametrisation_file;
  std::string optics_parametrisation_name;
  std::vector<Parametisation_aperture_configuration> inter_planes;
};

std::ostream & operator<<(std::ostream &s, const Parametisation_configuration &c);


struct Parametisation_aperture_configuration
{
  std::string to_marker_name;
  LHCApertureApproximator::aperture_type ap_type;
  double rect_rx, rect_ry, el_rx, el_ry;
  double displ_x, displ_y;
  int max_degree_x;
  int max_degree_tx;
  int max_degree_y;
  int max_degree_ty;
  double precision_x;
  double precision_tx;
  double precision_y;
  double precision_ty;

  bool common_terms;
};

std::ostream & operator<<(std::ostream &s, const Parametisation_aperture_configuration &c);


class MADParamGenerator
{
  public:
    int GenerateTrainingData(const Parametisation_configuration &conf);
    int GenerateDebugData(const Parametisation_configuration &conf);
    int GenerateGridDebugData(const Parametisation_configuration &conf);
    int GenerateXiContTDiscPhiContDebugData(const Parametisation_configuration &conf);
    int GenerateDiffractiveProtonsData(const Parametisation_configuration &conf);
    int GenerateTestingData(const Parametisation_configuration &conf);
    int GenerateApertureTestingData(const Parametisation_configuration &conf);
    int GenerateLostParticleData(const Parametisation_configuration &conf, int type=0);
    void BuildParametrization();

    void OpenXMLConfigurationFile(std::string file_name);

    void MakeAllParametrizations(bool generate_samples = true);
    void MakeParametrization(int id, bool generate_samples = true);
    void GenerateAllDebugSamples();
    void GenerateAllGridDebugSamples();
    void GenerateAllXiContTDiscPhiContDebugSamples();
    void GenerateAllDiffractiveProtons();
    void GenerateDebugSamples(int id);
    void GenerateGridDebugSamples(int id);
    void GenerateXiContTDiscPhiContDebugSamples(int id);
    void GenerateDiffractiveProtons(int id);
    void IdentifyAperturesForAll(int dist_type=0);
    void IdentifyApertures(int id, int dist_type=0);

    bool CheckParamConfId(int id);
    bool CheckApertureConfId(int param_id, int apreture_id);

    void TrainAndAddApertures(const Parametisation_configuration &conf, LHCOpticsApproximator &approximator, const char * f_out_name);

    Parametisation_configuration GetParamConfiguration(int id);
    Parametisation_aperture_configuration GetApertureConfiguration(int param_id, int apreture_id);

 // private:
    int BuildSample(const Parametisation_configuration &conf, std::string sample_file_name, bool recloss=false, bool compare_apert = false, int sample_type=0);
    int BuildDebugSample(const Parametisation_configuration &conf, std::string sample_file_name, bool recloss=false, bool compare_apert = false);
    int BuildGridDebugSample(const Parametisation_configuration &conf, std::string sample_file_name, bool recloss=false, bool compare_apert = false);
    int BuildXiContTDiscPhiContDebugSample(const Parametisation_configuration &conf, std::string sample_file_name, bool recloss=false, bool compare_apert = false);
    int BuildDiffractiveProtonsSample(const Parametisation_configuration &conf, std::string sample_file_name, bool recloss=false, bool compare_apert = false);

    void GenerateMADConfFile(const std::string &base_conf_file, const std::string &out_conf_file, const std::string &from_marker_name,
        double from_marker_s_pos, bool define_from, const std::string &to_marker_name, double to_marker_s_pos,
        bool define_to, int particles_number, bool aperture_limit=false,
        std::vector<std::string> scoring_planes = std::vector<std::string>(), const std::string &beam = std::string("lhcb1") );
    void GenerateRandomSamples(int number_of_particles, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    void GenerateDebugRandomSamples(int number_of_particles, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    void GenerateGridDebugRandomSamples(int number_of_particles, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    void GenerateXiContTDiscPhiContDebugRandomSamples(int number_of_particles, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    bool ComputeTheta(double beam_energy, double ksi, double t, double &theta);
    void GenerateDiffractiveProtonsSamples(int number_of_particles, double beam_energy, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    void GenerateElasticProtonsSamples(int number_of_particles, double beam_energy, double x_min, double x_max, double theta_x_min, double theta_x_max, double y_min, double y_max, double theta_y_min, double theta_y_max, double ksi_min, double ksi_max, const std::string &out_file_name);
    int AppendRootTree(std::string root_file_name, std::string out_prefix, std::string out_station, bool recloss, std::string lost_particles_tree_filename, const std::vector<std::string> &scoring_planes, bool compare_apert);  //return number of uppended entries
    void RunMAD(const std::string &conf_file);

    //auxiliary functions
    void Conf_file_processing(std::fstream &base_conf_file, std::fstream & conf_file, const std::string &from_marker_name, double from_marker_s_pos,
        bool define_from, const std::string &to_marker_name, double to_marker_s_pos, bool define_to, int particles_number, bool aperture_limit,
        const std::vector<std::string> &scoring_planes, const std::string &beam);
    std::string GetToken(std::fstream &base_conf_file);
    void ProcessToken(std::fstream &conf_file, const std::string &from_marker_name, double from_marker_s_pos, bool define_from,
        const std::string &to_marker_name, double to_marker_s_pos, bool define_to, int particles_number, bool aperture_limit,
        const std::string &token, const std::vector<std::string> &scoring_planes, const std::string &beam);
    TTree *CreateSamplesTree(TFile *f, std::string out_prefix, const std::vector<std::string> &scoring_planes);
    TTree *CreateAccelAcceptTree(TFile *f, std::string name = std::string("acc_acept_tree"));
    TTree *CreateLostParticlesTree(TFile *lost_particles_file);
    TTree *GetAccelAcceptTree(TFile *f);
    void WriteAccelAcceptTree(TFile *f, TTree *acc_acept_tree);

    Long64_t GetNumberOfEntries(std::string root_file_name, std::string out_prefix);
    Long64_t GetLostParticlesEntries(const Parametisation_configuration &conf);
    void ClearWorkingFiles(const Parametisation_configuration &conf, std::string sample_file_name);
    void DeleteLostParticlesFiles(const Parametisation_configuration &conf);
    void DeleteApertureTestFiles(const Parametisation_configuration &conf);
    void PrintTreeInfo(const Parametisation_configuration &conf, std::string sample_file_name);
    TTree *GetSamplesTree(const Parametisation_configuration &conf, std::string sample_file_name, TFile *&f);
    
    std::vector<std::string> GetAllScoringPlanes(int id);
    int RunMADXWithExternalData(int config_id, int size);

    private:
      RPXMLConfig xml_parser;
};


//execution in current directory


//base conf_file contains necessarily the following markers:
//#header_placement#
//#scoring_plane_definition#
//#start_point#
//#end_point#
//#scoring_plane_placement#
//#import_particles#
//#insert_particles#
//#output_mad_file#
