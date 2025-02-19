//
// $Id: Lepjets_Event.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Lepjets_Event.h
// Purpose: Represent a simple `event' consisting of leptons and jets.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// An instance of this class holds a list of `leptons' (as represented
// by Lepjets_Event_Lep) and `jets' (as represented by Lepjets_Event_Jet).
// We also record:
//
//   - Missing Et
//   - z-vertex
//   - run and event number
//
// CMSSW File      : interface/Lepjets_Event.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event.h

    @brief Represent a simple event consisting of lepton(s) and jet(s).

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

#ifndef HITFIT_LEPJETS_EVENT_H
#define HITFIT_LEPJETS_EVENT_H


#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Jet.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Lep.h"
#include <vector>
#include <iosfwd>


namespace hitfit {


/**
    @class Lepjets_Event

    @brief Represent a simple event consisting of lepton(s) and jet(s).
    An instance of this class holds a list of leptons (as represented by
    the Lepjets_Event_Lep class) and jets (as represented by Lepjets_Event_Jet
    class).  Also recorded are:
    - Missing transverse energy.
    -  \f$ z- \f$ vertex (Irrelevant for non-D0 experiment)
    - Run and event number (Irrelevant for non-D0 experiment)
 */
class Lepjets_Event
//
// Purpose: Represent a simple `event' consisting of leptons and jets.
//
{
public:

  // Constructor.
  /**
      @brief Constructor.

      @param runnum The run number.

      @param evnum The event number.
   */
  Lepjets_Event (int runnum, int evnum);

  // Get the run and event number.

  /**
     @brief Return a constant reference to the run number.
   */
  const int& runnum () const;

  /**
     @brief Return a reference to the run number.
   */
  int& runnum();

  /**
     @brief Return a constant reference to the event number.
   */
  const int& evnum () const;

  /**
     @brief Return a reference to the event number.
   */
  int& evnum();

  // Get the length of the lepton and jet lists.
  /**
     @brief Return the number of leptons in the event.
   */
  std::vector<Lepjets_Event_Lep>::size_type nleps () const;

  /**
     @brief Return the number of jets in the event.
   */
  std::vector<Lepjets_Event_Jet>::size_type njets () const;

  // Access leptons and jets.

  /**
     @brief Return a reference to lepton at index position <i>i</i>.

     @param i The lepton index position.
   */
  Lepjets_Event_Lep& lep (std::vector<Lepjets_Event_Lep>::size_type i);

  /**
     @brief Return a reference to jet at index position <i>i</i>.

     @param i The jet index position.
   */
  Lepjets_Event_Jet& jet (std::vector<Lepjets_Event_Jet>::size_type i);

  /**
     @brief Return a constant reference to lepton at index position <i>i</i>.

     @param i The lepton index position.
   */
  const Lepjets_Event_Lep& lep (std::vector<Lepjets_Event_Lep>::size_type i) const;

  /**
     @brief Return a constant reference to jet at index position <i>i</i>.

     @param i The jet index position.
   */
  const Lepjets_Event_Jet& jet (std::vector<Lepjets_Event_Jet>::size_type i) const;

  // Access missing Et.
  /**
     @brief Return a reference to the missing transverse energy.
   */
  Fourvec& met ();

  /**
     @brief Return a constant reference to the missing transverse energy.
   */
  const Fourvec& met () const;

  // Access kt resolution.

  /**
     @brief Return a reference to the  \f$ k_{T} \f$  resolution.
   */
  Resolution& kt_res ();

  /**
     @brief Return a const reference to the  \f$ k_{T} \f$  resolution.
   */
  const Resolution& kt_res () const;

  // Access the z-vertex.

  /**
     @brief Return the value of z-vertex.
   */
  double zvertex () const;

  /**
     @brief Return a reference to the value of z-vertex.
   */
  double& zvertex ();

  // Access the isMC flag.
  /**
     @brief Return the Monte Carlo flag.
   */
  bool isMC () const;

  /**
     @brief Set the Monte Carlo flag.
   */
  void setMC (bool isMC);

  // Access the discriminants.
  /**
     @brief Return a reference to the value of low-bias (LB) discriminant
     (Irrelevant for non-D0 experiment).
   */
  double& dlb ();

  /**
     @brief Return the value of low-bias (LB) discriminant
     (Irrelevant for non-D0 experiment).
   */
  double dlb () const;

  /**
     @brief Return a reference to the value of neural network (NN) discriminant
     (Irrelevant for non-D0 experiment).
   */
  double& dnn ();

  /**
     @brief Return a the value of neural network (NN) discriminant
     (Irrelevant for non-D0 experiment).
   */
  double dnn () const;

  // Sum all objects (leptons or jets) with type TYPE.
  /**
     @brief Return the sum of all objects' four-momentum which have
     a particular type.

     @param type The type code of the objects to be summed up.
   */
  Fourvec sum (int type) const;

  // Calculate kt --- sum of all objects plus missing Et.
  /**
     @brief Return the sum of all objects' four-momentum and
     missing transverse energy.
   */
  Fourvec kt () const;

  // Add new objects to the event.
  /**
     @brief Add a new lepton to the event.

     @param lep The lepton to be added.
   */
  void add_lep (const Lepjets_Event_Lep& lep);

  /**
     @brief Add a new jet to the event.

     @param jet The jet to be added.
   */
  void add_jet (const Lepjets_Event_Jet& jet);

  // Smear the objects in the event according to their resolutions.
  /**
     @brief Smear the objects in the event according to their resolutions.

     @param engine The underlying random number generator.

     @param smear_dir If <b>TRUE</b>, also smear the object's direction.<br>
     If <b>FALSE</b>, then only smear the magnitude of three-momentum.
   */
  void smear (CLHEP::HepRandomEngine& engine, bool smear_dir = false);

  // Sort according to pt.
  /**
     @brief Sort objects in the event according to their transverse momentum
      \f$ p_{T} \f$ .
   */
  void sort ();

  // Get jet types
  /**
     @brief Return the jet types in the event.
   */
  std::vector<int> jet_types() const;

  // Set jet types
  /**
     @brief Set the jet types in the event.
   */
  bool set_jet_types(const std::vector<int>&);

  // Remove objects failing pt and eta cuts.
  /**
     @brief Remove leptons which fail transverse momentum  \f$ p_{T} \f$ 
     and pseudorapidity  \f$ \eta \f$  cut.

     @param pt_cut Remove leptons which have transverse momentum  \f$ p_{T} \f$ 
     less than this value, in GeV.

     @param eta_cut Remove leptons which have absolute
     pseudorapidity  \f$ |\eta| \f$  more than this value.
   */
  int cut_leps (double pt_cut, double eta_cut);

  /**
     @brief Remove jets which fail transverse momentum  \f$ p_{T} \f$
     and pseudorapidity  \f$ \eta \f$  cut.

     @param pt_cut Remove jets which have transverse momentum  \f$ p_{T} \f$
     less than this value, in GeV.

     @param eta_cut Remove jetss which have absolute
     pseudorapidity  \f$ |\eta| \f$  more than this value.
   */
  int cut_jets (double pt_cut, double eta_cut);

  // Remove all but the first N jets.
  /**
     @brief Remove all but the first <i>n</i> jets.

     @param n The number of jets to keep.
   */
  void trimjets (std::vector<Lepjets_Event_Jet>::size_type n);

  // Dump this object.
  /**
     @brief Print the content of this object.

     @param s The output stream to which to write

     @param full If <b>TRUE</b>, print all information about this instance
     of Lepjets_Event.<br>
     If <b>FALSE</b>, print partial information about this instance
     of Lepjets_Event.
   */
  std::ostream& dump (std::ostream& s, bool full = false) const;

  /**
     @brief Return a string representing the jet permutation. The following
     notation is used for each type of jet:
     - g ISR/gluon.
     - b leptonic  \f$ b- \f$ quark.
     - B hadronic  \f$ b- \f$ quark.
     - w hadronic jet from  \f$ W- \f$ boson.
     - H  \f$ b- \f$ jet from Higgs boson.
     - ? Unknown.
   */
  std::string jet_permutation() const;

private:
  // The lepton and jet lists.

  /**
     The list of leptons in the event.
   */
  std::vector<Lepjets_Event_Lep> _leps;

  /**
     The list of jets in the event.
   */
  std::vector<Lepjets_Event_Jet> _jets;

  // Other event state.
  /**
     Missing transverse energy.
   */
  Fourvec _met;

  /**
     The  \f$ k_{T} \f$  resolution.
   */
  Resolution _kt_res;

  /**
     The  \f$ z- \f$ vertex of the event.
   */
  double _zvertex;

  /**
     The Monte Calro flag.
   */
  bool _isMC;

  /**
     The run number.
   */
  int _runnum;

  /**
     The event number.
   */
  int _evnum;

  /**
     The low-bias (LB) discriminant.
   */
  double _dlb;

  /**
     The neural network (NN) discriminant.
   */
  double _dnn;
};


// Print the object.
std::ostream& operator<< (std::ostream& s, const Lepjets_Event& ev);



} // namespace hitfit


#endif // not HITFIT_LEPJETS_EVENT_H

