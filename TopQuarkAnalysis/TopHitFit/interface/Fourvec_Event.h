//
// $Id: Fourvec_Event.h,v 1.2 2013/05/28 17:55:59 gartung Exp $
//
// File: hitfit/Fourvec_Event.h
// Purpose: Represent an event for kinematic fitting as a collection
//           of 4-vectors.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This class represents an `event' for kinematic fitting by
// Fourvec_Constrainer.  Each object in the event has the following
// attributes:
//
//   4-vector
//   mass
//     The kinematic fit assumes a fixed mass for each object.
//     That is specified by the `mass' attribute here.
//
//   p, phi, eta uncertainties
//   muon flag
//     If this is set, the `p' uncertainty is really in 1/p.
//
//   label
//     An integer that can be used to identify the object type.
//     I.e., lepton, b-jet from hadronic top, etc.
//
// There may be an object for a neutrino.
// If so, it is always at the end of the object list.
// It is not included in the count returned by nobjs() (but is included
// in nobjs_all()).
//
// We can also record one other `x' momentum, that will be added
// into the kt sum.  This can be used to store a missing Et that
// is not attributed to a neutrino (but is instead due to mismeasurement).
// Typically, this will be set to zero in events that have a neutrino,
// and to the measured missing Et in events that do not.
//
// CMSSW File      : interface/Fourvec_Event.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Fourvec_Event.h

    @brief Represent an event for kinematic fitting as a collection of
    four-momenta.

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
#ifndef HITFIT_FOURVEC_EVENT_H
#define HITFIT_FOURVEC_EVENT_H


#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <vector>
#include <iosfwd>


namespace hitfit {


/**
    @class FE_Obj

    @brief Represent a single object in a Fourvec_Event,
    this is just a dumb data container. Each object in a Fourvec_Event
    has the following attributes:

    - Four-momenta.
    - Mass. The kinematic fit assumes a fixed mass for each object,
    this is specified by the mass attribute.
    - Uncertainty in momentum \f$p\f$.
    - Uncertainty in azimuthal angle \f$\phi\f$.
    - Uncertainty in pseudorapidity \f$\eta\f$.
    - Muon/inverse momentum flag.  If the flag is set to true,
    then the uncertainty is interpreted as uncertainty in \f$1/p\f$ instead
    of \f$p\f$.
    - Label, an integer label that can be used to identify the object type:
    e.g. lepton, b-jet from hadronic top, etc.
 */
struct FE_Obj
//
// Purpose: Represent a single object in a Fourvec_Event.
//          This is just a dumb data container.
//
{
  // The 4-momentum of the object.
  /**
     The four-momentum of the object.
   */
  Fourvec p;

  // The mass of the object.
  // The kinematic fit will fix the mass to this.
  /**
     The mass of the object. The kinematic fitting algorithm will fix the
     mass to this.
   */
  double mass;

  // A label to identify the object type.
  /**
     An integer label to identify the object type.
   */
  int label;

  // p, phi, and eta uncertainties.
  /**
    Uncertainty in momentum \f$p\f$.
   */
  double p_error;

  /**
    Uncertainty in azimuthal angle \f$\phi\f$.
   */
  double phi_error;

  /**
    Uncertainty in pseudorapidity \f$\eta\f$.
   */
  double eta_error;

  // If this is true, then p_error is really an uncertainty in 1/p,
  // rather than p (and we should use 1/p as the fit variable).
  /**
     Muon/inverse momentum flag.  If the flag is set to TRUE,
     then the uncertainty is interpreted as uncertainty in \f$1/p\f$ instead
     of \f$p\f$.
   */
  bool muon_p;

  // Constructor, for convenience.
  /**
     @brief Constructor.

     @param the_p The four-momentum.

     @param the_mass The mass of the object.

     @param the_label An integer label to identify the object type.

     @param the_p_error Uncertainty in momentum \f$p\f$ or \f$1/p\f$
     if the muon flag is set to TRUE.

     @param the_phi_error Uncertainty in azimuthal angle \f$\phi\f$.

     @param the_eta_error Uncertainty in pseudorapidity \f$\eta\f$.

     @param the_muon_p Muon/inverse momentum flag.
  */
  FE_Obj (const Fourvec& the_p,
          double the_mass,
          int the_label,
          double the_p_error,
          double the_phi_error,
          double the_eta_error,
          bool the_muon_p);
};


// Print it out.
std::ostream& operator<< (std::ostream& s, const FE_Obj& o);


//************************************************************************


// The special label used for a neutrino.
/**
    A special label used for neutrino.
 */
const int nu_label = -1;


/**
    @brief Represent an event for kinematic fitting as a collection
    of four-momenta.  Each object is represented as an instance of FE_Obj.
    There may be an object for a neutrino.  If that is the case, it is always
    at the end of the object list.  It is not included in the count
    returned by nobjs().  But is is included in nobjs_all().

    We can also record the other \f$x\f$ momentum, that will be added into
    the \f$k_{T}\f$ sum. This can be used to store a missing transverse energy
    that is not attributed to a neutrino but is instead due to mismeasurement.
    Typically this will be set to zero in events that have a neutrino,
    and to the measured missing transverse energy in events that do not.
 */
class Fourvec_Event
//
// Purpose: Represent an event for kinematic fitting as a collection
//           of 4-vectors.
//
{
public:
  // Constructor.
  /**
     @brief Default constructor.
   */
  Fourvec_Event ();


  //****************************
  // Accessors.
  //

  // Return true if this event contains a neutrino.
  /**
     @brief Return TRUE is this event contains a neutrino, otherwise returns
      FALSE.
   */
  bool has_neutrino () const;

  // Return the number of objects in the event, not including any neutrino.
  /**
     @brief Return the number of objects in the event not including any
     neutrinos.
   */
  int nobjs () const;

  // Return the number of objects in the event, including any neutrino.
  /**
     @brief Return the number of objects in the event including any neutrinos.
   */
  int nobjs_all () const;

  // Access object I.  (Indexing starts with 0.)
  /**
     @brief Access object at index <i>i</i>, with the convention that the
     index starts at 0.

     @param i The index of the desired object.
   */
  const FE_Obj& obj (std::vector<FE_Obj>::size_type i) const;

  // Access the neutrino 4-momentum.
  /**
     @brief Access the neutrino four-momentum.
   */
  const Fourvec& nu () const;

  // Access the kt 4-momentum.
  /**
     @brief Access the \f$k_{T}\f$ four-momentum.
   */
  const Fourvec& kt () const;

  // Access the X 4-momentum.
  /**
     @brief Access the \f$x\f$ four-momentum.
   */
  const Fourvec& x () const;

  // Access the kt uncertainties.
  /**
     @brief Return the x uncertainty in \f$k_{T}\f$.
   */
  double kt_x_error () const;

  /**
     @brief Return the y uncertainty in \f$k_{T}\f$.
   */
  double kt_y_error () const;

  /**
     @brief Return the xy covariance in \f$k_{T}\f$.
   */
  double kt_xy_covar () const;

  // Print out the contents.
  friend std::ostream& operator<< (std::ostream& s,
                                   const Fourvec_Event& fe);


  //****************************
  // Modifiers.
  //

  // Add an object to the event.
  // (This should not be a neutrino --- use set_nu_p for that.)
  /**
     @brief Add an object to the event.  The object should not be a neutrino,
     use the method set_nu_p for that.

     @param obj The FE_Obj to add.
   */
  void add (const FE_Obj& obj);

  // Set the neutrino 4-momentum to P.
  // This adds a neutrino if there wasn't already one.
  /**
     @brief Set the neutrino four-momentum to \f$p\f$.  This method
     adds a neutrino if there wasn't already one.

     @param p The new four-momentum of the neutrino.
   */
  void set_nu_p (const Fourvec& p);

  // Set the 4-momentum of object I to P.
  /**
     @brief Set the four-momentum of object at index <i>i</i> to \f$p\f$.

     @param i The position index of the object to change.

     @param p The new four-momentum of object at index <i>i</i>.
   */
  void set_obj_p (std::vector<FE_Obj>::size_type i, const Fourvec& p);

  // Set the 4-momentum of the X object.
  /**
     @brief Set the four-momentum of the \f$x\f$ object.

     @param p The new four-momentum of the \f$x\f$ object.
   */
  void set_x_p (const Fourvec& p);

  // Set the kt uncertainties.
  /**
     @brief Set the uncertainties on \f$k_{T}\f$.

     @param kt_x_error The uncertainty in the \f$x-\f$component of \f$k_{T}\f$.

     @param kt_y_error The uncertainty in the \f$y-\f$component of \f$k_{T}\f$.

     @param kt_xy_covar The covariance between the \f$x-\f$ and
     \f$y-\f$component of \f$k_{T}\f$.

   */
  void set_kt_error (double kt_x_error, double kt_y_error, double kt_xy_covar);


private:
  // The list of contained objects.
  /**
     The list of contained objects in the event.
   */
  std::vector<FE_Obj> _objs;

  // Cached kt.  This should always be equal to the sum of all the
  // object momenta, including x.
  /**
     Cached \f$k_{T}\f$, this should always be equal to the sum of all the
     object momenta, including \f$x\f$.
   */
  Fourvec _kt;

  // Momemtum of the X object.
  /**
     Four-momentum of the \f$x\f$ object.
   */
  Fourvec _x;

  // The kt uncertainties.
  /**
     The uncertainty in the \f$x-\f$component of \f$k_{T}\f$.   */
  double _kt_x_error;

  /**
     The uncertainty in the \f$y-\f$component of \f$k_{T}\f$.   */
  double _kt_y_error;

  /**
     The covariance between the \f$x-\f$ and
     \f$y-\f$component of \f$k_{T}\f$.
  */
  double _kt_xy_covar;

  // Flag that a neutrino has been added.
  /**
     Flag that a neutrino has been added to the event.
   */
  bool _has_neutrino;
};


} // namespace hitfit


#endif // not HITFIT_FOURVEC_EVENT_H
