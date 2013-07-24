//
// $Id: Top_Fit.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Top_Fit.h
// Purpose: Handle jet permutations.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/Top_Fit.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Top_Fit.h

    @brief Handle and fit jet permutations of an event.  This is the
    primary interface between user's Lepjets_Event and HitFit kinematic
    fitting algorithm.

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

#ifndef HITFIT_TOP_FIT_H
#define HITFIT_TOP_FIT_H


#include "TopQuarkAnalysis/TopHitFit/interface/Constrained_Top.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>


namespace hitfit {


class Lepjets_Event;
class Fit_Results;


//
// Indices for the various results lists we make in Fit_Results.
//
/**
    Indices for the various results lists we store in Fit_Results.
 */
enum Lists
{
  all_list = 0,         // All events.
  noperm_list = 1,      // All jet assignments are correct.
  semicorrect_list = 2, // Jets assigned to the correct top.
  limited_isr_list = 3, // Top three jets are not ISR.
  topfour_list = 4,     // Top four jets are not ISR, any other jets are ISR.
  btag_list = 5,        // All tagged jets were assigned as b-jets.
  htag_list = 6,        // All tagged jets were assigned as b-jets or higgs.
  n_lists = 7
};


//*************************************************************************


/**
    @brief Hold on to parameters for the Top_Fit class.
*/
class Top_Fit_Args
//
// Purpose: Hold on to parameters for the Top_Fit class.
//
//   bool print_event_flag - If true, print the event after the fit.
//   bool do_higgs_flag    - If true, fit ttH events.
//   double jet_mass_cut   - Reject events with jet masses larger than this.
//   double mwhad_min_cut  - Reject events with hadronic W mass before
//                           fitting smaller than this.
//   double mwhad_max_cut  - Reject events with hadronic W mass before
//                           fitting larger than this.
//   double mtdiff_max_cut - Reject events where the difference between
//                           leptonic and hadronic top masses before fitting
//                           is larger than this.
//   int nkeep             - Number of results to keep in each list.
//   bool solve_nu_tmass   - If true, use hadronic top mass to constrain
//                           the neutrino pz.  Otherwise constrain the
//                           neutrino + lepton mass to W mass.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  /**
     @brief Constructor, initialize an instance of Top_Fit_Args
     from an instance of Defaults object.

     @param defs The Defaults instance from which to initialize.  The
     instance must contain the following parameters with types and names:
     - bool <i>print_event_flag</i>.
     - bool <i>do_higgs_flag</i>.
     - double <i>jet_mass_cut</i>.
     - double <i>mwhad_min_cut</i>.
     - double <i>mwhad_max_cut</i>.
     - double <i>mtdiff_max_cut</i>.
     - int <i>nkeep</i>.
     - bool <i>solve_nu_tmass</i>.

   */
  Top_Fit_Args (const Defaults& defs);

  // Retrieve parameter values.
  /**
     @brief Return the <i>print_event_flag</i> parameter.
   */
  bool print_event_flag () const;

  /**
     @brief Return the <i>do_higgs_flag</i> parameter.
   */
  bool do_higgs_flag () const;

  /**
     @brief Return the <i>jet_mass_cut</i> parameter.
   */
  double jet_mass_cut () const;

  /**
     @brief Return the <i>mwhad_min_cut</i> parameter.
   */
  double mwhad_min_cut () const;

  /**
     @brief Return the <i>mwhad_min_cut</i> parameter.
   */
  double mwhad_max_cut () const;

  /**
     @brief Return the <i>mwhad_max_cut</i> parameter.
   */
  double mtdiff_max_cut () const;

  /**
     @brief Return the <i>nkeep</i> parameter.
   */
  int nkeep () const;

  /**
     @brief Return the <i>solve_nu_tmass</i> parameter.
   */
  bool solve_nu_tmass() const;

  // Arguments for subobjects.
  const Constrained_Top_Args& constrainer_args () const;


private:
  // Hold on to parameter values.
  /**
     If <b>TRUE</b>, then print the event after the fit.  Otherwise don't
     print.
   */
  bool _print_event_flag;

  /**
     If <b>TRUE</b>, then fit a  \f$ t\bar{t}H \f$  events. Otherwise fit
      \f$ t\bar{t} \f$  event.
   */
  bool _do_higgs_flag;

  /**
     Reject event before fit if there is at least one jet which have mass
     larger than this value, in GeV.
   */
  double _jet_mass_cut;

  /**
     Reject event before fit if the mass of the hadronic  \f$ W- \f$ boson
     is smaller than this value, in GeV.
   */
  double _mwhad_min_cut;

  /**
     Reject event before fit if the mass of the hadronic  \f$ W- \f$ boson
     is larger than this value, in GeV.
   */
  double _mwhad_max_cut;

  /**
     Reject event before fit if the mass difference between leptonic top
     and hadronic top is larger than this value, in GeV.
   */
  double _mtdiff_max_cut;

  /**
     The number of Fit_Results from different jet permutations to keep.
     It is recommended that the number is set accordingly to the maximally
     allowed number of jets in the event.  The number for possible
     permutations,  \f$ N_{t\bar{t}} \f$ , as a function of number of jets,
      \f$ n \f$ , for  \f$ t\bar{t} \f$  event is given by:
     \f[
     N_{t\bar{t}}(n) = \frac{n!}{(n-4)!};~ n \ge 4
     \f]
     The number for possible permutations,  \f$ N_{t\bar{t}H} \f$ , as a
     function of number of jets,  \f$ n \f$ , for  \f$ t\bar{t}H \f$  is given by:
     \f[
     N_{t\bar{t}}(n) = \frac{n!}{(n-6)!2!};~ n \ge 6
     \f]

   */
  int _nkeep;

  /**
     If <b>TRUE</b>, then solve the neutrino longitudinal  \f$ z- \f$ component
     by requiring the leptonic top have the same mass as the hadronic top.
     If <b>FALSE</b>, then solve the neutrino longitudinal  \f$ z- \f$ component
     by requiring the lepton and neutrino mass to equal to the mass of
     the  \f$ W- \f$ boson  \f$ m_{W} \f$ .
   */
  bool _solve_nu_tmass;

  /**
     The internal state, parameter settings for the Constrained_Top instance
     within an instance of Top_Fit.
   */
  Constrained_Top_Args _args;

};


//*************************************************************************

/**
    @brief Handle and fit jet permutations of an event.  This is the
    primary interface between user's Lepjets_Event and HitFit kinematic
    fitting algorithm.
*/
class Top_Fit
//
// Purpose: Handle jet permutations.
//
{
public:
  // Constructor.
  // LEPW_MASS, HADW_MASS, and TOP_MASS are the masses to which
  // those objects should be constrained.  To remove a constraint,
  // set the mass to 0.
  /**
     @brief Constructor.

     @param args The parameter settings.

     @param lepw_mass The mass to which the leptonic  \f$ W- \f$ boson should be
     constrained to.  A value of zero means this constraint will be removed.

     @param hadw_mass The mass to which the hadronic  \f$ W- \f$ boson should be
     constrained to.  A value of zero means this constraint will be removed.

     @param top_mass The mass to which the top quark should be constrained to.
     A value of zero means this constraint will be removed.
   */
  Top_Fit (const Top_Fit_Args& args,
           double lepw_mass,
           double hadw_mass,
           double top_mass);

  // Fit a single jet permutation.  Return the results for that fit.
  /**
      @brief Fit for a single jet permutation.

      @param ev Input: The event to fit, Output: the event after the fit.

      @param nuz Input: A flag to indicate which neutrino solution to be used.
      <br>
      <b>FALSE</b> means use solution with smaller absolute value.<br>
      <b>TRUE</b> means use solution with larger absolute value.

      @param umwhad The mass of hadronic  \f$ W- \f$ boson before the fit.

      @param utmass The mass of the top quarks before fitting, averaged from
      the values of leptonic and hadronic top quark mass.

      @param mt The mass of the top quark after fitting.

      @param sigmt The uncertainty of the mass of the top quark after fitting.

      @param pullx Pull quantities for well-measured variables.

      @param pully Pull quantities for poorly-measured variables.
   */
  double fit_one_perm (Lepjets_Event& ev,
                       bool& nuz,
                       double& umwhad,
                       double& utmass,
                       double& mt,
                       double& sigmt,
                       Column_Vector& pullx,
                       Column_Vector& pully);

  // Fit all jet permutations in EV.
  /**
     @brief Fit all jets permutations in ev.  This function returns
     a Fit_Results object, which is not easy to extract information from.
     Users are recommended to use the class RunHitFit as interface to fit
     all permutations of all event.

     @param ev Input: The event to fit, Output: the event after the fit.
   */
  Fit_Results fit (const Lepjets_Event& ev);

  // Print.
  friend std::ostream& operator<< (std::ostream& s, const Top_Fit& fitter);

  /**
     @brief Return a constant reference to the fit arguments.
   */
  const Top_Fit_Args& args() const;

private:
  // The object state.
  const Top_Fit_Args _args;
  Constrained_Top _constrainer;
  double _lepw_mass;
  double _hadw_mass;
};


} // namespace hitfit


#endif // not HITFIT_TOP_FIT_H
