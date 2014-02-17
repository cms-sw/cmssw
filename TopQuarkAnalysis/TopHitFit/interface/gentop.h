//
// $Id: gentop.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/gentop.h
// Purpose: Toy ttbar event generator for testing.
// Created: Jul, 2000, sss.
//
// This is a very simple event generator for ttbar events, to allow some
// basic tests of the mass fitting code.  We generate random ttbars,
// with kinematics pulled out of a hat, and then decay them into l+jets
// events.  No radiation or other such luxuries, and, as mentioned, any
// kinematic distribuions will certainly be wrong.  But the generated
// events should satisfy the l+jets mass constraints.
//
// CMSSW File      : interface/gentop.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file gentop.h

    @brief A toy event generator for
    \f$t\bar{t} \to \ell + 4~\mathrm{jets}\f$ events.

    This is a very simple event generator for
    \f$t\bar{t} \to \ell + 4~\mathrm{jets}\f$ events, to allow some
    basic tests of the kinematic fitting code.  The code generates random
    \f$t\bar{t}\f$ events with kinematics pulled out of a hat (a random
    number generator), and then decay them into
    \f$t\bar{t} \to \ell + 4~\mathrm{jets}\f$ events.  No physics behind the
    generation except for four-momentum conservation and
    energy-mass-momentum relation.  No luxuries such as radiation etc, and,
    as mentioned, any kinematic distribution will certainly not corresponding
    to physical reality.  But the generated events should satisfy the
    \f$\ell + 4~\mathrm{jets}\f$ mass constraints, and therefore would be
    usable for testing.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */

#include <string>
#include <iosfwd>

#include "CLHEP/Random/Random.h"


namespace hitfit {


class Defaults;
class Lepjets_Event;


/**
    @class Gentop_Args.

    @brief Hold on to parameters for the toy event generator.
 */
class Gentop_Args
//
// Hold on to parameters for the toy event generator.
//   float mt           - Generated top mass.
//   float sigma_mt     - Width of top mass distribution.
//
//   float mh           - Generated Higgs mass.
//   float sigma_mh     - Width of Higgs mass distribution.
//
//   float mw           - Generated W mass.
//   float sigma_mw     - Width of W mass distribution.
//
//   float mb           - Generated b mass.
//   float sigma_mb     - Width of b mass distribution.
//
//   float t_pt_mean    - Mean pt of the generated top quarks.
//                        (It will be drawn from an exponential distribution.)
//   float recoil_pt_mean-Mean pt of ttbar system.
//                        (It will be drawn from an exponential distribution.)
//   float boost_sigma  - Width of z-boost of ttbar system.
//   float m_boost      - Mass of z-boost of ttbar system.
//   float sxv_tageff   - Assumed efficiency of SVX b-tag.
//   bool smear         - If true, smear the event.
//   bool smear_dir     - If false, smear only energies, not directions.
//   bool muon          - If false, decay leptonic ts into electrons.
//                        Otherwise, decay into muons.
//   string ele_res_str - Electron resolution, for Vector_Resolution.
//   string muo_res_str - Muon resolution, for Vector_Resolution.
//   string jet_res_str - Jet resolution, for Vector_Resolution.
//   string kt_res_str  - Kt resolution, for Resolution.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  /**
     @brief Constructor, initialize an instance of Gentop_Args from
     an instance of Defaults object.

     @param defs The defaults instance from which to initialize.
     The instance must contain the following parameters with types
     and names:
     - double <i>t_pt_mean</i>.
     - double <i>mt</i>.
     - double <i>sigma_mt</i>.
     - double <i>mh</i>.
     - double <i>sigma_mh</i>.
     - double <i>recoil_pt_mean</i>.
     - double <i>boost_sigma</i>.
     - double <i>m_boost</i>.
     - double <i>mb</i>.
     - double <i>sigma_mb</i>.
     - double <i>mw</i>.
     - double <i>sigma_mw</i>.
     - double <i>svx_tageff</i>.
     - bool <i>smear</i>.
     - bool <i>smear_dir</i>.
     - bool <i>muon</i>.
     - string <i>ele_res_str</i>.
     - string <i>muo_res_str</i>.
     - string <i>jet_res_str</i>.
     - string <i>kt_res_str</i>.
   */
  Gentop_Args (const Defaults& defs);

  // Retrieve parameter values.

  /**
     @brief Return the value of <i>t_pt_mean</i> parameter.
   */
  double t_pt_mean () const;

  /**
     @brief Return the value of <i>mt</i> parameter.
   */
  double mt () const;

  /**
     @brief Return the value of <i>sigma_mt</i> parameter.
   */
  double sigma_mt () const;

  /**
     @brief Return the value of <i>mh</i> parameter.
   */
  double mh () const;

  /**
     @brief Return the value of <i>sigma_mh</i> parameter.
   */
  double sigma_mh () const;

  /**
     @brief Return the value of <i>recoil_pt_mean</i> parameter.
   */
  double recoil_pt_mean () const;

  /**
     @brief Return the value of <i>boost_sigma</i> parameter.
   */
  double boost_sigma () const;

  /**
     @brief Return the value of <i>m_boost</i> parameter.
   */
  double m_boost () const;

  /**
     @brief Return the value of <i>mb</i> parameter.
   */
  double mb () const;

  /**
     @brief Return the value of <i>sigma_mb</i> parameter.
   */
  double sigma_mb () const;

  /**
     @brief Return the value of <i>mw</i> parameter.
   */
  double mw () const;

  /**
     @brief Return the value of <i>sigma_mw</i> parameter.
   */
  double sigma_mw () const;

  /**
     @brief Return the value of <i>svx_tageff</i> parameter.
   */
  double svx_tageff () const;

  /**
     @brief Return the value of <i>smear</i> parameter.
   */
  bool smear () const;

  /**
     @brief Return the value of <i>smear_dir</i> parameter.
   */
  bool smear_dir () const;

  /**
     @brief Return the value of <i>muon</i> parameter.
   */
  bool muon () const;

  /**
     @brief Return the value of <i>ele_res_str</i> parameter.
   */
  std::string ele_res_str () const;

  /**
     @brief Return the value of <i>muon_res_str</i> parameter.
   */
  std::string muo_res_str () const;

  /**
     @brief Return the value of <i>jet_res_str</i> parameter.
   */
  std::string jet_res_str () const;

  /**
     @brief Return the value of <i>kt_res_str</i> parameter.
   */
  std::string  kt_res_str () const;

private:
  // Hold on to parameter values.

  /**
     Mean transverse momentum \f$p_{T}\f$ of the generated top quarks, in GeV,
     drawn from an exponential distribution.
   */
  double _t_pt_mean;

  /**
     Mass of the generated top quark, in GeV.
   */
  double _mt;

  /**
     Width of the generated top quark mass distribution, in GeV.
  */
  double _sigma_mt;

  /**
     Mass of the generated Higgs boson, in GeV.
   */
  double _mh;

  /**
     Width of the generated Higgs boson mass distribution, in GeV.
   */
  double _sigma_mh;

  /**
     Mean transverse momentum \f$p_{T}\f$ of the generated \f$t\bar{t}\f$
     system, in GeV, drawn from an exponential distribution.
   */
  double _recoil_pt_mean;

  /**
     Width of the \f$z-\f$boost of the \f$t\bar{t}\f$ system, in GeV.
   */
  double _boost_sigma;

  /**
     Mass of the \f$z-\f$boost of the \f$t\bar{t}\f$ system, in GeV.
   */
  double _m_boost;

  /**
     Mass of the generated <i>b</i> quark, in GeV.
   */
  double _mb;

  /**
     Width of the generated <i>b</i> quark mass distribution, in GeV.
   */
  double _sigma_mb;

  /**
     Mass of the generated <i>W</i> boson, in GeV.
   */
  double _mw;

  /**
     Width of the generated <i>W</i> boson mass, in GeV.
   */
  double _sigma_mw;

  /**
     Assumed efficiency of SVX (Secondary Vertex) b-tagging, for most cases
     it is irrelevant.
   */
  double _svx_tageff;

  /**
     If TRUE, smear the event.<br>
     If FALSE, don't smear the event.
   */
  bool   _smear;

  /**
     If TRUE, smear the energy and direction of individual particles.<br>
     If FALSE, only smear the energy of individual particle.
   */
  bool   _smear_dir;

  /**
     If TRUE, decay the leptonic top quark into muon.<br>
     If FALSE, decay the leptonic top quark into electron.
  */
  bool   _muon;

  /**
     Electron resolution information in format suitable for Vector_Resolution.
   */
  std::string _ele_res_str;

  /**
     Muon resolution information in format suitable for Vector_Resolution.
   */
  std::string _muo_res_str;

  /**
     Jet resolution information in format suitable for Vector_Resolution.
   */
  std::string _jet_res_str;

  /**
     \f$k_{T}\f$ resolution information in format suitable for
     Vector_Resolution.
   */
  std::string _kt_res_str;
};


// Generate a ttbar -> ljets event.
/**
    @brief Generate a \f$t\bar{t} \to \ell + 4~\mathrm{jets}\f$ event.

    @param args The parameter settings for this event.

    @param engine The underlying random number generator.
 */
Lepjets_Event gentop (const Gentop_Args& args,
                      CLHEP::HepRandomEngine& engine);

// Generate a ttH -> ljets+bb event.
/**
    @brief Generate a \f$t\bar{t}H \to \ell + b\bar{b} + 4~\mathrm{jets}\f$
    event.

    @param args The parameter settings for this event.

    @param engine The underlying random number generator.
 */
Lepjets_Event gentth (const Gentop_Args& args,
                      CLHEP::HepRandomEngine& engine);


} // namespace hitfit
