//
// $Id: Lepjets_Event_Jet.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Lepjets_Event_Jet.h
// Purpose: Represent a `jet' in a Lepjets_Event.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This is like Lepjets_Event_Jet, except that we store some
// additional information:
//
//   - svx tag flag
//   - slt tag flag
//   -   slt lepton 4-vector
//   -   slt lepton energy deposition
//
// CMSSW File      : interface/Lepjets_Event_Jet.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event_Jet.h

    @brief Represent a jet in an instance of Lepjets_Event class.

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

#ifndef HITFIT_LEPJETS_EVENT_JET_H
#define HITFIT_LEPJETS_EVENT_JET_H


#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Lep.h"
#include <iosfwd>


namespace hitfit {


/**
    Possible types of jet in an instance of Lepjets_Event class.
 */
enum Jet_Labels {
  isr_label = 0,
  lepb_label = 11,
  hadb_label = 12,
  hadw1_label = 13,
  hadw2_label = 14,
  higgs_label = 15,
  unknown_label = 20
};


/**
    @class Lepjets_Event_Jet

    @brief A class to represent a jet in an instance of Lepjets_Event class.
    The class is derived from the Lepjets_Event_Lep class.  In addition
    to the information stored in Lepjets_Event_Lep class, this class
    holds the following information:
    - SVX (Secondary Vertex) tag flag (Irrelevant for non-D0 experiment).
    - SLT (Soft Lepton Tag) flag (Irrelevant for non-D0 experiment).
    - SLT lepton four-momentum (Irrelevant for non-D0 experiment).
    - SLT lepton energy deposition (Irrelevant for non-D0 experiment).
 */
class Lepjets_Event_Jet
  : public Lepjets_Event_Lep
//
// Purpose: Represent a `jet' in a Lepjets_Event.
//
{
public:
  // Constructor.
  /**
     @brief Constructor.

     @param p The four-momemtum.

     @param type The type code.

     @param res The jet resolution.

     @param svx_tag Boolean flag for SVX tag.

     @param slt_tag Boolean flag for SLT tag.

     @param tag_lep The SLT lepton four-momentum.

     @param slt_edep The SLT lepton energy deposition.
   */
  Lepjets_Event_Jet (const Fourvec& p,
                     int type,
                     const Vector_Resolution& res,
                     bool svx_tag = false,
                     bool slt_tag = false,
                     const Fourvec& tag_lep = Fourvec(),
                     double slt_edep = 0);

  // Access the svx tag flag.
  /**
     Return the SVX tag flag.
   */
  bool svx_tag () const;

  /**
     Return a reference to the SVX tag flag.
   */
  bool& svx_tag ();

  // Access the slt tag flag.
  /**
     Return the SLT tag flag.
   */

  bool slt_tag () const;
  /**
     Return a reference to the SLT tag flag.
   */
  bool& slt_tag ();

  // Access the tag lepton four-momentum.
  /**
     Return a reference to the SLT lepton.
   */
  Fourvec& tag_lep ();

  /**
     Return a constant reference to the SLT lepton.
   */
  const Fourvec& tag_lep () const;

  // Access the tag lepton energy deposition.
  /**
     Return the SLT lepton energy deposition.
   */
  double slt_edep () const;
  /**
     Return a reference to SLT lepton energy deposition.
   */
  double& slt_edep ();

  // Access the uncorrected jet energy.
  /**
     Return the uncorrected jet energy.
   */
  double e0 () const;

  /**
     Return a reference of the uncorrected jet energy.
   */
  double& e0 ();

  // Print the content of this object.
  /**
     @brief Print the content of this object.

     @param s The output stream to which to write

     @param full If <b>TRUE</b>, print all information about this instance
     of Lepjets_Event_Lep.<br>
     If <b>FALSE</b>, print partial information about this instance
     of Lepjets_Event_Lep.
   */
  std::ostream& dump (std::ostream& s, bool full = false) const;


private:
  // The object state.
  /**
     Boolean flag for the SVX tag.
   */
  bool _svx_tag;

  /**
     Boolean flag for the SLT tag.
   */
  bool _slt_tag;

  /**
     The SLT lepton four-momentum.
   */
  Fourvec _tag_lep;

  /**
     The SLT lepton energy deposition.
   */
  double _slt_edep;

  /**
     The uncorrected jet energy.
   */
  double _e0;
};


// Print this object.
std::ostream& operator<< (std::ostream& s, const Lepjets_Event_Jet& ev);

// Helper function to translate jet type from integer to char/string
/**
    @brief Helper function: Translate jet type code from integer to char.
    The following notation is used for each type of jet:
    - g ISR/gluon.
    - b leptonic  \f$ b- \f$ quark.
    - B hadronic  \f$ b- \f$ quark.
    - w hadronic jet from  \f$ W- \f$ boson.
    - H  \f$ b- \f$ jet from Higgs boson.
    - ? Unknown.

    @param type The jet type code
 */
char jetTypeChar(int type);

/**
    @brief Helper function: Translate jet type code from integer to char.
    The following notation is used for each type of jet:
    - g ISR/gluon.
    - b leptonic  \f$ b- \f$ quark.
    - B hadronic  \f$ b- \f$ quark.
    - w hadronic jet from  \f$ W- \f$ boson.
    - H  \f$ b- \f$ jet from Higgs boson.
    - ? Unknown.

    @param type The jet type code
 */
std::string jetTypeString(int type);

/**
    @brief Helper function: Translate jet type code from a list of numbers
    to a string.
    - g ISR/gluon.
    - b leptonic  \f$ b- \f$ quark.
    - B hadronic  \f$ b- \f$ quark.
    - w hadronic jet from  \f$ W- \f$ boson.
    - H  \f$ b- \f$ jet from Higgs boson.
    - ? Unknown.

    @param jet_types The jet type codes in vector form.
 */
template<class T>
std::string
jetTypeString(std::vector<T> jet_types)
{


    std::ostringstream ret;

    for (size_t j = 0 ; j != jet_types.size() ; ++j) {
        ret << jetTypeChar((int) (jet_types[j]));
    }

    return ret.str();
}


} // namespace hitfit


#endif // not HITFIT_LEPJETS_EVENT_JET_H

