//
// $Id: Lepjets_Event_Lep.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Lepjets_Event_Lep.h
// Purpose: Represent a `lepton' in a Lepjets_Event class.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// For each lepton, we store:
//
//   - 4-momentum
//   - type code
//   - Vector_Resolution
//
// CMSSW File      : interface/Lepjets_Event_Lep.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event_Lep.h

    @brief Represent a lepton in an instance of Lepjets_Event class.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    July 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */

#ifndef HITFIT_LEPJETS_EVENT_LEP_H
#define HITFIT_LEPJETS_EVENT_LEP_H


#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"
#include <iosfwd>


namespace hitfit {

/**
    Possible types of lepton in an instance of Lepjets_Event class.
 */
enum Lepton_Labels {
  lepton_label = 1,  // generic lepton
  electron_label = 2,
  muon_label = 3
};


/**
    @brief Represent a lepton in an instance of Lepjets_Event class. This class
    hold the following information:
    - Four-momentum.
    - The type code (electron, muon, or generic lepton).
    - The resolution in Vector_Resolution type.
 */
class Lepjets_Event_Lep
//
// Purpose: Represent a `lepton' in a Lepjets_Event class.
//
{
public:
  // Constructor.
  /**
     @brief Constructor, create a new instance of Lepjets_Event_Lep.

     @param p The four-momentum.

     @param type The type code.

     @param res The resolution.
   */
  Lepjets_Event_Lep (const Fourvec& p,
                     int type,
                     const Vector_Resolution& res);

  // Access the 4-momentum.
  /**
     @brief Return a reference to the four-momentum.
   */
  Fourvec& p ();

  /**
     @brief Return a constant reference to the four-momentum.
   */
  const Fourvec& p () const;

  // Access the type code.
  /**
     @brief Return a reference to the type code.
   */
  int& type ();

  /**
     @brief Return the type code.
   */
  int type () const;

  // Access the resolution.
  /**
     @brief Return a constant reference to the resolution.
   */
  const Vector_Resolution& res () const;

  /**
     @brief Return a reference to the resolution.
   */
  Vector_Resolution& res ();

  // Return resolutions for this object.
  /**
     @brief Return the uncertainty in momentum \f$p\f$ or \f$p_{T}\f$
     (\f$1/p\f$ or \f$1/p_{T}\f$ if the lepton is a tracking object).
   */
  double p_sigma () const;

  /**
     @brief Return the uncertainty in pseudorapidity \f$\eta\f$.
   */
  double eta_sigma () const;

  /**
     @brief Return the uncertainty in azimuthal angle \f$\phi\f$.
   */
  double phi_sigma () const;

  // Smear this object.
  // If SMEAR_DIR is false, smear the momentum only.
  /**
     @brief Smear this object.

     @param engine The underlying random number generator.

     @param smear_dir If <b>TRUE</b>, also smear the object's direction.<br>
     If <b>FALSE</b>, then only smear the magnitude of three-momentum.
   */
  void smear (CLHEP::HepRandomEngine& engine, bool smear_dir = false);

  // Dump out this object.
  /**
     @brief Print the content of this object.

     @param s The output stream to which to write.

     @param full If <b>TRUE</b>, print all information about this instance
     of Lepjets_Event_Lep.<br>
     If <b>FALSE</b>, print partial information about this instance
     of Lepjets_Event_Lep.
   */
  std::ostream& dump (std::ostream& s, bool full = false) const;

  // Sort on pt.
  /**
     @brief Comparison operator for sorting purpose, based on
     \$p_{T}\$.

     @param x The other instance of Lepjets_Event to be compared.
   */
  bool operator< (const Lepjets_Event_Lep& x) const;


private:
  // The object state.

  /**
     The four-momentum.
   */
  Fourvec _p;

  /**
     The type code.
   */
  int _type;

  /**
     The resolution.
   */
  Vector_Resolution _res;
};


// Print the object.
std::ostream& operator<< (std::ostream& s, const Lepjets_Event_Lep& ev);


} // namespace hitfit


#endif // not HITFIT_LEPJETS_EVENT_LEP_H

