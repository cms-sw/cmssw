#ifndef TotemRPProtonTransportParametrization_LHC_OPTICS_APPROXIMATOR_H
#define TotemRPProtonTransportParametrization_LHC_OPTICS_APPROXIMATOR_H

#include <string>
#include "TNamed.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include <memory>
#include "TMatrixD.h"

#include "SimTransport/TotemRPProtonTransportParametrization/interface/TMultiDimFet.h"

struct MadKinematicDescriptor {
  double x;        ///< m
  double theta_x;  ///< rad
  double y;        ///< m
  double theta_y;  ///< rad
  double ksi;      ///< 1
};

class LHCApertureApproximator;

/**
 *\brief Class finds the parametrisation of MADX proton transport and transports the protons according to it
 * 5 phase space variables are taken in to configuration: x, y, theta_x, theta_y, xi
 * xi < 0 for momentum losses (that is for diffractive protons)
**/
class LHCOpticsApproximator : public TNamed {
public:
  LHCOpticsApproximator();
  /// begin and end position along the beam of the particle to transport, training_tree, prefix of data branch in the tree
  LHCOpticsApproximator(std::string name,
                        std::string title,
                        TMultiDimFet::EMDFPolyType polynom_type,
                        std::string beam_direction,
                        double nominal_beam_momentum);
  LHCOpticsApproximator(const LHCOpticsApproximator &org);
  const LHCOpticsApproximator &operator=(const LHCOpticsApproximator &org);

  enum polynomials_selection { AUTOMATIC, PREDEFINED };
  enum beam_type { lhcb1, lhcb2 };
  void Train(TTree *inp_tree,
             std::string data_prefix = std::string("def"),
             polynomials_selection mode = PREDEFINED,
             int max_degree_x = 10,
             int max_degree_tx = 10,
             int max_degree_y = 10,
             int max_degree_ty = 10,
             bool common_terms = false,
             double *prec = nullptr);
  void Test(TTree *inp_tree,
            TFile *f_out,
            std::string data_prefix = std::string("def"),
            std::string base_out_dir = std::string(""));
  void TestAperture(TTree *in_tree,
                    TTree *out_tree);  ///< x, theta_x, y, theta_y, ksi, mad_accepted, parametriz_accepted

  double ParameterOutOfRangePenalty(double par_m[], bool invert_beam_coord_sytems = true) const;

  /// Basic 3D transport method
  /// MADX canonical variables
  /// IN/OUT: (x, theta_x, y, theta_y, xi) [m, rad, m, rad, 1]
  /// returns true if transport possible
  /// if theta is calculated from momentum p, use theta_x = p.x() / p.mag() and theta_y = p.y() / p.mag()
  bool Transport(const double *in,
                 double *out,
                 bool check_apertures = false,
                 bool invert_beam_coord_sytems = true) const;
  bool Transport(const MadKinematicDescriptor *in,
                 MadKinematicDescriptor *out,
                 bool check_apertures = false,
                 bool invert_beam_coord_sytems = true) const;  //return true if transport possible

  /// Basic 2D transport method
  /// MADX canonical variables
  /// IN : (x, theta_x, y, theta_y, xi) [m, rad, m, rad, 1]
  /// OUT : (x, y) [m, m]
  /// returns true if transport possible
  bool Transport2D(const double *in,
                   double *out,
                   bool check_apertures = false,
                   bool invert_beam_coord_sytems = true) const;

  bool Transport_m_GeV(double in_pos[3],
                       double in_momentum[3],
                       double out_pos[3],
                       double out_momentum[3],
                       bool check_apertures,
                       double z2_z1_dist) const;  ///< pos, momentum: x,y,z;  pos in m, momentum in GeV/c

  void PrintInputRange();
  bool CheckInputRange(const double *in, bool invert_beam_coord_sytems = true) const;
  void AddRectEllipseAperture(
      const LHCOpticsApproximator &in, double rect_x, double rect_y, double r_el_x, double r_el_y);
  void PrintOpticalFunctions();
  void PrintCoordinateOpticalFunctions(TMultiDimFet &parametrization,
                                       const std::string &coord_name,
                                       const std::vector<std::string> &input_vars);

  /**
     *\brief returns linearised transport matrix for x projection
     * |  dx_out/dx_in    dx_out/dthx_in   |
     * | dthx_out/dx_in  dthx_out/dthx_in  |
     *
     * input:  [m], [rad], xi:-1...0
     */
  void GetLineariasedTransportMatrixX(double mad_init_x,
                                      double mad_init_thx,
                                      double mad_init_y,
                                      double mad_init_thy,
                                      double mad_init_xi,
                                      TMatrixD &tr_matrix,
                                      double d_mad_x = 10e-6,
                                      double d_mad_thx = 10e-6);

  /**
     *\brief returns linearised transport matrix for y projection
     * |  dy_out/dy_in    dy_out/dthy_in   |
     * | dthy_out/dy_in  dthy_out/dthy_in  |
     *
     * input:  [m], [rad], xi:-1...0
     */
  void GetLineariasedTransportMatrixY(double mad_init_x,
                                      double mad_init_thx,
                                      double mad_init_y,
                                      double mad_init_thy,
                                      double mad_init_xi,
                                      TMatrixD &tr_matrix,
                                      double d_mad_y = 10e-6,
                                      double d_mad_thy = 10e-6);

  double GetDx(double mad_init_x,
               double mad_init_thx,
               double mad_init_y,
               double mad_init_thy,
               double mad_init_xi,
               double d_mad_xi = 0.001);
  double GetDxds(double mad_init_x,
                 double mad_init_thx,
                 double mad_init_y,
                 double mad_init_thy,
                 double mad_init_xi,
                 double d_mad_xi = 0.001);
  inline beam_type GetBeamType() const { return beam; }

  /// returns linear approximation of the transport parameterization
  /// takes numerical derivatives (see parameter ep) around point `atPoint' (this array has the same structure as `in' parameter in Transport method)
  /// the linearized transport: x = Cx + Lx*theta_x + vx*x_star
  void GetLinearApproximation(double atPoint[],
                              double &Cx,
                              double &Lx,
                              double &vx,
                              double &Cy,
                              double &Ly,
                              double &vy,
                              double &D,
                              double ep = 1E-5);

private:
  void Init();
  double s_begin_;  ///< begin of transport along the reference orbit
  double s_end_;    ///< end of transport along the reference orbit
  beam_type beam;
  double nominal_beam_energy_;                  ///< GeV
  double nominal_beam_momentum_;                ///< GeV/c
  bool trained_;                                ///< trained polynomials
  std::vector<TMultiDimFet *> out_polynomials;  //! pointers to polynomials
  std::vector<std::string> coord_names;
  std::vector<LHCApertureApproximator> apertures_;  ///< apertures on the way

#ifndef __CINT_
  friend class ProtonTransportFunctionsESSource;
#endif  // __CINT__

  TMultiDimFet x_parametrisation;        ///< polynomial approximation for x
  TMultiDimFet theta_x_parametrisation;  ///< polynomial approximation for theta_x
  TMultiDimFet y_parametrisation;        ///< polynomial approximation for y
  TMultiDimFet theta_y_parametrisation;  ///< polynomial approximation for theta_y

  //train_mode mode_;  //polynomial selection mode - selection done by fitting function or selection from the list according to the specified order
  enum class VariableType { X, THETA_X, Y, THETA_Y };
  //internal methods
  void InitializeApproximators(polynomials_selection mode,
                               int max_degree_x,
                               int max_degree_tx,
                               int max_degree_y,
                               int max_degree_ty,
                               bool common_terms);
  void SetDefaultAproximatorSettings(TMultiDimFet &approximator, VariableType var_type, int max_degree);
  void SetTermsManually(TMultiDimFet &approximator, VariableType variable, int max_degree, bool common_terms);

  void AllocateErrorHists(TH1D *err_hists[4]);
  void AllocateErrorInputCorHists(TH2D *err_inp_cor_hists[4][5]);
  void AllocateErrorOutputCorHists(TH2D *err_out_cor_hists[4][5]);

  void DeleteErrorHists(TH1D *err_hists[4]);
  void DeleteErrorCorHistograms(TH2D *err_cor_hists[4][5]);

  void FillErrorHistograms(double errors[4], TH1D *err_hists[4]);
  void FillErrorDataCorHistograms(double errors[4], double var[5], TH2D *err_cor_hists[4][5]);

  void WriteHistograms(TH1D *err_hists[4],
                       TH2D *err_inp_cor_hists[4][5],
                       TH2D *err_out_cor_hists[4][5],
                       TFile *f_out,
                       std::string base_out_dir);

  ClassDef(LHCOpticsApproximator, 1)  // Proton transport approximator
};

class LHCApertureApproximator : public LHCOpticsApproximator {
public:
  enum class ApertureType { NO_APERTURE, RECTELLIPSE };

  LHCApertureApproximator();
  LHCApertureApproximator(const LHCOpticsApproximator &in,
                          double rect_x,
                          double rect_y,
                          double r_el_x,
                          double r_el_y,
                          ApertureType type = ApertureType::RECTELLIPSE);

  bool CheckAperture(const double *in, bool invert_beam_coord_sytems = true) const;  //x, thx. y, thy, ksi
  //bool CheckAperture(MadKinematicDescriptor *in);  //x, thx. y, thy, ksi
private:
  double rect_x_, rect_y_, r_el_x_, r_el_y_;
  ApertureType ap_type_;

  ClassDef(LHCApertureApproximator, 1)  // Aperture approximator
};
#endif  //TotemRPProtonTransportParametrization_LHC_OPTICS_APPROXIMATOR_H
