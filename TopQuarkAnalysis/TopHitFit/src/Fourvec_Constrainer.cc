//
// $Id: Fourvec_Constrainer.cc,v 1.2 2012/11/21 19:07:26 davidlt Exp $
//
// File: src/Fourvec_Constrainer.cc
// Purpose: Do a kinematic fit for a set of 4-vectors, given a set
//          of mass constraints.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Fourvec_Constrainer.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**

    @file Fourvec_Constrainer.cc

    @brief Do a kinematic fit for a set of four-vectors, given a set
    of mass constraints. Contains definitions of helper class
    Fourvec_Constraint_Calculator and some helper functions.
    See the documentation for the header file Fourvec_Constrainer.h for
    details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Pair_Table.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Chisq_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <cmath>
#include <iostream>


using std::sqrt;
using std::exp;
using std::cos;
using std::sin;
using std::ostream;
using std::vector;


namespace hitfit {


//*************************************************************************
// Argument handling.
//


Fourvec_Constrainer_Args::Fourvec_Constrainer_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _use_e (defs.get_bool ("use_e")),
    _e_com (defs.get_float ("e_com")),
    _ignore_met (defs.get_bool ("ignore_met")),
    _chisq_constrainer_args (defs)
{
}


bool Fourvec_Constrainer_Args::use_e () const
//
// Purpose: Return the use_e parameter.
//          See the header for documentation.
//
{
  return _use_e;
}


double Fourvec_Constrainer_Args::e_com () const
//
// Purpose: Return the e_com parameter.
//          See the header for documentation.
//
{
  return _e_com;
}


bool Fourvec_Constrainer_Args::ignore_met () const
//
// Purpose: Return the ignore_met parameter.
//          See the header for documentation.
//
{
  return _ignore_met;
}


const Chisq_Constrainer_Args&
Fourvec_Constrainer_Args::chisq_constrainer_args () const
//
// Purpose: Return the contained subobject parameters.
//
{
  return _chisq_constrainer_args;
}


//*************************************************************************
// Variable layout.
//
// We need to map the quantities we fit onto the vectors of well- and
// poorly-measured quantities.
//

//
// The well-measured variables consist of three variables for each
// object.  If we are using transverse momentum constraints,
// these fill be followed by the two cartesian components of kt.
//
// Each object is represented by three variables: the momentum (or 1/p
// if the muon flag was set), and the two spherical angles, phi and eta.
// Here is how they're ordered.
//
/**
    Offset indices for the component of four-momentum variables.
 */
typedef enum {
  p_offs = 0,
  phi_offs = 1,
  eta_offs = 2
} Offsets;

/**
    Offset indices for the components of missing transverse energy
    (or \f$k_{T}\f$) variables.
 */
typedef enum {
  x_offs = 0,
  y_offs = 1
} Kt_Offsets;

//
// If there is a neutrino, then it is at index 1 of the poorly-measured
// set (otherwise, that set is empty).
//
/**
    If there is a neutrino, then it is at index 1 of the vector of
    poorly-measured variables.
 */
typedef enum {
  nu_z = 1
} Unmeasured_Variables;



namespace {


/**
    @brief Helper function: Return the starting variable index for object
    number <i>i</i>.

    @param i The object's index.
 */
int obj_index (int i)
//
// Purpose: Return the starting variable index for object I.
//
// Inputs:
//   i -           The object index.
//
// Returns:
//   The index in the well-measured set of the first variable
//   for object I.
//
{
  return i*3 + 1;
}


} // unnamed namespace


//*************************************************************************
// Object management.
//


Fourvec_Constrainer::Fourvec_Constrainer (const Fourvec_Constrainer_Args& args)
//
// Purpose: Constructor.
//
// Inputs:
//   args -        The parameter settings for this instance.
//
  : _args (args)
{
}


void Fourvec_Constrainer::add_constraint (std::string s)
//
// Purpose: Specify an additional constraint S for the problem.
//          The format for S is described in the header.
//
// Inputs:
//   s -           The constraint to add.
//
{
  _constraints.push_back (Constraint (s));
}


void Fourvec_Constrainer::mass_constraint (std::string s)
//
// Purpose: Specify the combination of objects that will be returned by
//          constrain() as the mass.  The format of S is the same as for
//          normal constraints.  The LHS specifies the mass to calculate;
//          the RHS should be zero.
//          This should only be called once.
//
// Inputs:
//   s -           The constraint defining the mass.
//
{
  assert (_mass_constraint.size() == 0);
  _mass_constraint.push_back (Constraint (s));
}


/**
    @brief Output stream operator, print the content of this
    Fourvec_Constrainer to an output stream.

    @param s The output stream to which to write.

    @param c The instance of Fourvec_Constrainer to be printed.

*/
std::ostream& operator<< (std::ostream& s, const Fourvec_Constrainer& c)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   c -           The object to write.
//
// Returns:
//   The stream S.
//
{
  s << "Constraints: (e_com = " << c._args.e_com() << ") ";
  if (c._args.use_e())
    s << "(E)";
  s << "\n";

  for (std::vector<Constraint>::size_type i=0; i < c._constraints.size(); i++)
    s << "  " << c._constraints[i] << "\n";

  if (c._mass_constraint.size() > 0) {
    s << "Mass constraint:\n";
    s << c._mass_constraint[0] << "\n";
  }
  return s;
}


//*************************************************************************
// Event packing and unpacking.
//


namespace {


/**
    @brief Helper function: For all objects in the Fourvec_Event instance
    <i>ev</i>, adjust their four-momenta to have their requested masses.

    @param ev The event on which to operate.

    @param use_e_flag If TRUE, keep the energy and scale the
    three-momentum.<br>
    If FALSE, keep the three-momentum and scale the energy.
 */
void adjust_fourvecs (Fourvec_Event& ev,
                      bool use_e_flag)
//
// Purpose: For all objects in EV, adjust their 4-momenta
//          to have their requested masses.
//
// Inputs:
//   ev -          The event on which to operate.
//   use_e_flag -  If true, keep E and scale 3-momentum.
//                 If false, keep the 3-momentum and scale E.
//
{
  int nobjs = ev.nobjs ();
  for (int i=0; i < nobjs; i++) {
    const FE_Obj& obj = ev.obj (i);
    Fourvec p = obj.p;
    if (use_e_flag)
      adjust_p_for_mass (p, obj.mass);
    else
      adjust_e_for_mass (p, obj.mass);
    ev.set_obj_p (i, p);
  }
}


/**
    @brief Helper function: Convert object at index <i>ndx</i> from its
    representation in the vector of well-measured variables <i>c</i>
    to a four-momentum.

    @param c The vector of well-measured variables.

    @param ndx The index of the object in which we are interested.

    @param obj The object from the instance of Fourvec_Event.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.
 */
Fourvec get_p_eta_phi_vec (const Column_Vector& c,
                           int ndx,
                           const FE_Obj& obj,
                           bool use_e_flag)
//
// Purpose: Convert object NDX from its representation in the set
//          of well-measured variables C to a 4-vector.
//
// Inputs:
//   c -           The vector of well-measured variables.
//   ndx -         The index of the object in which we're interested.
//   obj -         The object from the Fourvec_Event.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//
// Returns:
//   The object's 4-momentum.
//
{
  // Get the energy and momentum of the object.
  double e, p;

  if (use_e_flag) {
    // We're using E as a fit variable.  Get it directly.
    e = c(ndx + p_offs);

    // Take into account the muon case.
    if (obj.muon_p) e = 1/e;

    // Find the momentum given the energy.
    if (obj.mass == 0)
      p = e;
    else {
      double xx = e*e - obj.mass*obj.mass;
      if (xx >= 0)
    p = sqrt (xx);
      else
    p = 0;
    }
  }
  else {
    // We're using P as a fit variable.  Fetch it.
    p = c(ndx + p_offs);

    // Take into account the muon case.
    if (obj.muon_p) p = 1/p;

    // Find the energy given the momentum.
    e = (obj.mass == 0 ? p : sqrt (obj.mass*obj.mass + p*p));
  }

  // Get angular variables.
  double phi = c(ndx + phi_offs);
  double eta = c(ndx + eta_offs);
  if (fabs (eta) > 50) {
    // Protect against ridiculously large etas
    eta = eta > 0 ? 50 : -50;
  }
  double exp_eta = exp (eta);
  double iexp_eta = 1/exp_eta;
  double sin_theta = 2 / (exp_eta + iexp_eta);
  double cos_theta = (exp_eta - iexp_eta) / (exp_eta + iexp_eta);

  // Form the 4-momentum.
  return Fourvec (p * sin_theta * cos (phi),
                  p * sin_theta * sin (phi),
                  p * cos_theta,
                  e);
}


/**
    @brief Helper function: Initialize the variables in the vector of
    well-measured variables <i>c</i> describing object at index <i>ndx</i>
    from its Fourvec_Event representation <i>obj</i>.

    @param obj The object from the instance of Fourvec_Event.

    @param c The vector of well-measured variables.

    @param ndx The index of the object in which we are interested.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.
 */
void set_p_eta_phi_vec (const FE_Obj& obj,
                        Column_Vector& c,
                        int ndx,
                        bool use_e_flag)
//
// Purpose: Initialize the variables in the well-measured set C describing
//          object NDX from its Fourvec_Event representation OBJ.
//
// Inputs:
//   obj -         The object from the Fourvec_Event.
//   c -           The vector of well-measured variables.
//   ndx -         The index of the object in which we're interested.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//
//
{
  if (use_e_flag)
    c(ndx + p_offs) = obj.p.e();
  else
    c(ndx + p_offs) = obj.p.vect().mag();

  if (obj.muon_p) c(ndx + p_offs) = 1/c(ndx + p_offs);
  c(ndx + phi_offs) = obj.p.phi();
  c(ndx + eta_offs) = obj.p.pseudoRapidity();
}


/**
    @brief Helper function: Construct a 3 by 3 error matrix for an object.

    @param p_sig The uncertainty on the momentum \f$p\f$.

    @param phi_sig The uncertainty on the azimuthal angle \f$\phi\f$.

    @param eta_sig The uncertainty on the pseudorapidity \f$\eta\f$.
 */
Matrix error_matrix (double p_sig,
                     double phi_sig,
                     double eta_sig)
//
// Purpose: Set up the 3x3 error matrix for an object.
//
// Inputs:
//   p_sig -       The momentum uncertainty.
//   phi_sig -     The phi uncertainty.
//   eta_sig -     The eta uncertainty.
//
// Returns:
//   The object's error matrix.
//
{
  Matrix err (3, 3, 0);
  err(1+  p_offs, 1+  p_offs) = p_sig * p_sig;
  err(1+phi_offs, 1+phi_offs) = phi_sig * phi_sig;
  err(1+eta_offs, 1+eta_offs) = eta_sig * eta_sig;
  return err;
}


/**
    @brief Helper function: Take the information from a
    Fourvec_Event instance <i>ev</i> and pack it into vectors
    of well- and poorly-measured variables.  Also set up
    the error matrices.

    @param ev The event to pack.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param use_kt_flag If TRUE, then we are also pack \f$k){T}\f$
    variables.<br>
    If FALSE, then we are not.

    @param xm The vector of well-measured variables.

    @param ym The vector of poorly-measured variables.

    @param G_i The error matrix for well-measured variables.

    @param Y Inverse error matrix for poorly-measured variables.
 */
void pack_event (const Fourvec_Event& ev,
                 bool use_e_flag,
                 bool use_kt_flag,
                 Column_Vector& xm,
                 Column_Vector& ym,
                 Matrix& G_i,
                 Diagonal_Matrix& Y)
//
// Purpose: Take the information from a Fourvec_Event EV and pack
//          it into vectors of well- and poorly-measured variables.
//          Also set up the error matrices.
//
// Inputs:
//   ev -          The event to pack.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//   use_kt_flag - True if we're to pack kt variables.
//
// Outputs:
//   xm -          Vector of well-measured variables.
//   ym -          Vector of poorly-measured variables.
//   G_i -         Error matrix for well-measured variables.
//   Y -           Inverse error matrix for poorly-measured variables.
//
{
  // Number of objects in the event.
  int nobjs = ev.nobjs ();

  int n_measured_vars = nobjs * 3;
  if (use_kt_flag)
    n_measured_vars += 2;

  // Clear the error matrix.
  G_i = Matrix (n_measured_vars, n_measured_vars, 0);

  // Loop over objects.
  for (int i=0; i<nobjs; i++) {
    const FE_Obj& obj = ev.obj (i);
    int this_index = obj_index (i);
    set_p_eta_phi_vec (obj, xm, this_index, use_e_flag);
    G_i.sub (this_index, this_index, error_matrix (obj.p_error,
                                                   obj.phi_error,
                                                   obj.eta_error));

  }

  if (use_kt_flag) {
    // Set up kt.
    int kt_ndx = obj_index (nobjs);
    xm (kt_ndx+x_offs) = ev.kt().x();
    xm (kt_ndx+y_offs) = ev.kt().y();

    // And its error matrix.
    G_i(kt_ndx+x_offs, kt_ndx+x_offs) = ev.kt_x_error() * ev.kt_x_error();
    G_i(kt_ndx+y_offs, kt_ndx+y_offs) = ev.kt_y_error() * ev.kt_y_error();
    G_i(kt_ndx+x_offs, kt_ndx+y_offs) = ev.kt_xy_covar();
    G_i(kt_ndx+y_offs, kt_ndx+x_offs) = ev.kt_xy_covar();
  }

  // Handle a neutrino.
  if (ev.has_neutrino()) {
    ym(nu_z) = ev.nu().z();
    Y = Diagonal_Matrix (1, 0);
  }
}


/**
    @brief Helper function: Update the content of <i>ev</i> from the variable
    sets <i>x</i> and <i>y</i>.

    @param ev The event to update.

    @param x Vector of well-measured variables.

    @param y Vector of poorly-measured variables.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param use_kt_flag If TRUE, then we are also pack \f$k){T}\f$
    variables.<br>
    If FALSE, then \f$k){T}\f$ variables are not pack.
 */
void unpack_event (Fourvec_Event& ev,
                   const Column_Vector& x,
                   const Column_Vector& y,
                   bool use_e_flag,
                   bool use_kt_flag)
//
// Purpose: Update the contents of EV from the variable sets X and Y.
//
// Inputs:
//   ev -          The event.
//   x -           Vector of well-measured variables.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//   use_kt_flag - True if we're too unpack kt variables.
//
// Outputs:
//   ev -          The event after updating.
//
{
  // Do all the objects.
  Fourvec sum = Fourvec(0);
  for (int j=0; j<ev.nobjs(); j++) {
    const FE_Obj& obj = ev.obj (j);
    ev.set_obj_p (j, get_p_eta_phi_vec (x, obj_index (j), obj, use_e_flag));
    sum += obj.p;
  }


  if (use_kt_flag) {
    int kt_ndx = obj_index (ev.nobjs());
    Fourvec kt = Fourvec (x(kt_ndx+x_offs), x(kt_ndx+y_offs), 0, 0);
    Fourvec nu = kt - sum;
    if (ev.has_neutrino()) {
      nu.setPz (y(nu_z));
      adjust_e_for_mass (nu, 0);
      ev.set_nu_p (nu);
    }
    else {
      adjust_e_for_mass (nu, 0);
      ev.set_x_p (nu);
    }
  }
}


} // unnamed namespace


//*************************************************************************
// Constraint evaluation.
//


namespace {


/**
    @brief Helper function: Compute the dot product
    \f$\mathbf{v}_{1} \cdot \mathbf{v}_{2}\f$ and its gradient w.r.t.
    \f$p\f$, \f$\phi\f$, and \f$\theta\f$ of each four-vector.

    @param v1 The first four-vector in the dot product.

    @param v2 The second four-vector in the dot product.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param v1_x Gradients of the dot product w.r.t. to
    \f$p\f$, \f$\phi\f$, and \f$\theta\f$ of \f$\mathbf{v}_{1}\f$.

    @param v2_x Gradients of the dot product w.r.t. to
    \f$p\f$, \f$\phi\f$, and \f$\theta\f$ of \f$\mathbf{v}_{2}\f$.

    @param badflag Return status, set to TRUE for singular cases
    when calculating the dot product or gradient (zero values for
    any of these: energy \f$E\f$, magnitude of three-momentum \f$p\f$,
    magnitude of transverse component of three-momentum \f$p_{T}\f$.

    @par Returns:
    The dot product.
 */
double dot_and_gradient (const Fourvec& v1,
                         const Fourvec& v2,
                         bool use_e_flag,
                         double v1_x[3],
                         double v2_x[3],
                         bool& badflag)
//
// Purpose: Compute the dot product v1.v2 and its gradients wrt
//          p, phi, and theta of each 4-vector.
//
// Inputs:
//   v1 -          The first 4-vector in the dot product.
//   v2 -          The second 4-vector in the dot product.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//
// Outputs:
//   v1_x -        Gradients of the dot product wrt v1's p, phi, theta.
//   v2_x -        Gradients of the dot product wrt v2's p, phi, theta.
//   badflag -     Set to true for the singular case (vectors vanish).
//
// Returns:
//   The dot product.
//
{
  // Calculate the dot product.
  double dot = v1 * v2;

  double p1 = v1.vect().mag();
  double p2 = v2.vect().mag();
  double e1 = v1.e();
  double e2 = v2.e();
  double pt1 = v1.vect().perp();
  double pt2 = v2.vect().perp();

  // Protect against the singular case.
  badflag = false;
  if (p1 == 0 || p2 == 0 || e1 == 0 || e2 == 0 || pt1 == 0 || pt2 == 0) {
    badflag = true;
    v1_x[p_offs] = v1_x[phi_offs] = v1_x[eta_offs] = 0;
    v2_x[p_offs] = v2_x[phi_offs] = v2_x[eta_offs] = 0;
    return false;
  }

  // Calculate the gradients.
  v1_x[p_offs] = (dot - v1.m2() * e2 / e1) / p1;
  v2_x[p_offs] = (dot - v2.m2() * e1 / e2) / p2;

  if (use_e_flag) {
    v1_x[p_offs] *= e1 / p1;
    v2_x[p_offs] *= e2 / p2;
  }

  v1_x[phi_offs] = v1(1)*v2(0) - v1(0)*v2(1);
  v2_x[phi_offs] = -v1_x[phi_offs];

  double fac = v1(0)*v2(0) + v1(1)*v2(1);
  v1_x[eta_offs] = pt1*v2(2) - v1(2)/pt1 * fac;
  v2_x[eta_offs] = pt2*v1(2) - v2(2)/pt2 * fac;

  return dot;
}


/**
    @brief Helper function: Tally up dot product gradients for an object into
    <i>Bx</i>, the gradients of the well-measured variables.

    @param constraint_no The number/index of the constraint.

    @param base_index The index in the vector of well-measured variables
    of the first variable for this object.

    @param sign The sign with which these gradients should be added into
    <i>Bx</i>, either \f$+1\f$ or \f$-1\f$ (that is, which side of
    the constraint equation).

    @param grad The gradient for this object w.r.t. to \f$p\f$,
    \f$\phi\f$, and \f$\theta\f$.

    @param Bx The gradients of well-measured variables.

    @par Output:
    - <i>Bx</i>.
 */
void addin_obj_gradient (int constraint_no,
                         int sign,
                         int base_index,
                         const double grad[],
                         Matrix& Bx)
//
// Purpose: Tally up the dot product gradients for an object
//          into Bx.
//
// Inputs:
//   constraint_no-The number of the constraint.
//   base_index -  The index in the well-measured variable list
//                 of the first variable for this object.
//   sign -        The sign with which these gradients should be
//                 added into Bx, either +1 or -1.  (I.e., which
//                 side of the constraint equation.)
//   grad -        The gradients for this object, vs. p, phi, theta.
//   Bx -          The well-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//
{
  Bx(base_index + p_offs,   constraint_no) += sign * grad[p_offs];
  Bx(base_index + phi_offs, constraint_no) += sign * grad[phi_offs];
  Bx(base_index + eta_offs, constraint_no) += sign * grad[eta_offs];
}


/**

    @brief Helper function: Tally up the dot product gradients for a neutrino
    into <i>Bx</i> and <i>By</i>, the gradient vector of well-measured and
    poorly-measured variables, respectively.

    @param constraint_no The number/index of the constraint.

    @param sign The sign with which these gradients should be added into
    <i>Bx</i>, either \f$+1\f$ or \f$-1\f$ (that is, which side of
    the constraint equation).

    @param kt_ndx The index of the \f$k_{T}\f$ variables in the variables
    array.

    @param grad The gradient for this object w.r.t. to \f$p\f$,
    \f$\phi\f$, and \f$\theta\f$.

    @param Bx The gradients of well-measured variables.

    @param By The gradients of poorly-measured variables.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.
    - <i>By</i> The updated gradients of the poorly-measured variables.

 */
void addin_nu_gradient (int constraint_no,
                        int sign,
                        int kt_ndx,
                        const double grad[],
                        Matrix& Bx, Matrix& By)
//
// Purpose: Tally up the dot product gradients for a neutrino
//          into Bx and By.
//
// Inputs:
//   constraint_no-The number of the constraint.
//   sign -        The sign with which these gradients should be
//                 added into Bx, either +1 or -1.  (I.e., which
//                 side of the constraint equation.)
//   kt_ndx -      The index of the kt variables in the variables array.
//   grad -        The gradients for this object, vs. p, phi, theta.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//   By -          The poorly-measured variable gradients, updated.
//
{
  Bx(kt_ndx+x_offs,constraint_no) += sign*grad[p_offs];  // Really p for now.
  Bx(kt_ndx+y_offs,constraint_no) += sign*grad[phi_offs];// Really phi for now.
  By(nu_z,         constraint_no) += sign*grad[eta_offs]; // Really theta ...
}


/**
    @brief Helper function: Tally up the dot product gradients for an object
    (which may or may not be a neutrino) into <i>Bx</i> and <i>By</i>.

    @param ev The event we are fitting.

    @param constraint_no The number/index of the constraint.

    @param sign The sign with which these gradients should be added into
    <i>Bx</i>, either \f$+1\f$ or \f$-1\f$ (that is, which side of
    the constraint equation).

    @param obj_no The number of the object.

    @param grad The gradient for this object w.r.t. to \f$p\f$,
    \f$\phi\f$, and \f$\theta\f$.

    @param Bx The gradients of the well-measured variables.

    @param By The gradients of the poorly-measured variables.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.<br>
    - <i>By</i> The updated gradients of the poorly-measured variables.

 */
void addin_gradient (const Fourvec_Event& ev,
                     int constraint_no,
                     int sign,
                     int obj_no,
                     const double grad[],
                     Matrix& Bx,
                     Matrix& By)
//
// Purpose: Tally up the dot product gradients for an object (which may
//          or may not be a neutrino) into Bx and By.
//
// Inputs:
//   ev -          The event we're fitting.
//   constraint_no-The number of the constraint.
//   sign -        The sign with which these gradients should be
//                 added into Bx, either +1 or -1.  (I.e., which
//                 side of the constraint equation.)
//   obj_no -      The number of the object.
//   grad -        The gradients for this object, vs. p, phi, theta.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//   By -          The poorly-measured variable gradients, updated.
//
{
  if (obj_no >= ev.nobjs()) {
    assert (obj_no == ev.nobjs());
    addin_nu_gradient (constraint_no, sign, obj_index (obj_no), grad, Bx, By);
  }
  else
    addin_obj_gradient (constraint_no, sign, obj_index (obj_no), grad, Bx);
}


/**
    @brief Helper function: Tally up the gradients from a single dot product
    into <i>Bx</i> and <i>By</i>.

    @param ev The event we are fitting.

    @param constraint_no The number/index of the constraint.

    @param sign The sign with which these gradients should be added into
    <i>Bx</i>, either \f$+1\f$ or \f$-1\f$ (that is, which side of
    the constraint equation).

    @param i The number/index of the first object.

    @param igrad The gradients of the first object w.r.t. to \f$p\f$,
    \f$\phi\f$, and \f$\theta\f$.

    @param j The number/index of the second object.

    @param jgrad The gradients of the second object w.r.t. to \f$p\f$,
    \f$\phi\f$, and \f$\theta\f$.

    @param Bx The gradients of the well-measured variables.

    @param By The gradients of the poorly-measured variables.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.<br>
    - <i>By</i> The updated gradients of the poorly-measured variables.
 */
void addin_gradients (const Fourvec_Event& ev,
                      int constraint_no,
                      int sign,
                      int i,
                      const double igrad[],
                      int j,
                      const double jgrad[],
                      Matrix& Bx,
                      Matrix& By)
//
// Purpose: Tally up the gradients from a single dot product into Bx and By.
//
// Inputs:
//   ev -          The event we're fitting.
//   constraint_no-The number of the constraint.
//   sign -        The sign with which these gradients should be
//                 added into Bx, either +1 or -1.  (I.e., which
//                 side of the constraint equation.)
//   i -           The number of the first object.
//   igrad -       The gradients for the first object, vs. p, phi, theta.
//   j -           The number of the second object.
//   jgrad -       The gradients for the second object, vs. p, phi, theta.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//   By -          The poorly-measured variable gradients, updated.
//
{
  addin_gradient (ev, constraint_no, sign, i, igrad, Bx, By);
  addin_gradient (ev, constraint_no, sign, j, jgrad, Bx, By);
}


/**
    @brief Helper function: Add the \f$m^{2}\f$ into the constraint values.

    @param ev The event we are fitting.

    @param constraints The list of constraints.

    @param F Vector of constraint values.

    @par Output:
    - <i>F</i> The updated vector of constraint values.
 */
void add_mass_terms (const Fourvec_Event& ev,
                     const vector<Constraint>& constraints,
                     Row_Vector& F)
//
// Purpose: Add the m^2 terms into the constraint values.
//
// Inputs:
//   ev -          The event we're fitting.
//   constraints - The list of constraints.
//   F -           Vector of constraint values.
//
// Outputs:
//   F -           Vector of constraint values, updated.
//
{
  for (std::vector<Constraint>::size_type i=0; i<constraints.size(); i++)
    F(i+1) += constraints[i].sum_mass_terms (ev);
}


/**
    @brief Helper function: Calculate the mass constraints and gradients.
    At this sage, the gradients are calculated not quite with respect to
    the fit variables, instead for all objects (including neutrino) we
    calculate the gradients with respect to \f$p\f$, \f$\phi\f$, and
    \f$\theta\f$.  They will be converted via the appropriate
    Jacobian transformation later

    @param ev The event we are fitting.

    @param pt Table of cached pair assigments.

    @param constraints The list of constraints.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param F The vector of constraint values, should be passed with the
    correct dimensionaly.

    @param Bx The gradients of the well-measured variables, should be passed
    with the correct dimensionaly.

    @param By The gradients of the poorly-measured variables, should be
    passed with the scorrect dimensionaly.

    @par Output:
    - <i>F</i> The vector of constraint variables.<br>
    - <i>Bx</i> The updated gradients of the well-measured variables.<br>
    - <i>By</i> The updated gradients of the poorly-measured variables.
 */
bool calculate_mass_constraints (const Fourvec_Event& ev,
                                 const Pair_Table& pt,
                                 const vector<Constraint>& constraints,
                                 bool use_e_flag,
                                 Row_Vector& F,
                                 Matrix& Bx,
                                 Matrix& By)
//
// Purpose: Calculate the mass constraints and gradients.
//          Note: At this stage, the gradients are calculated not
//                quite with respect to the fit variables; instead, for
//                all objects (including the neutrino) we calculate
//                the gradients with respect to p, phi, theta.  They'll
//                be converted via appropriate Jacobian transformations
//                later.
//
// Inputs:
//   ev -          The event we're fitting.
//   pt -          Table of cached pair assignments.
//   constraints - The list of constraints.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//
// Outputs:
//   (nb. these should be passed in correctly dimensioned.)
//   F -           Vector of constraint values.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
{
  int npairs = pt.npairs ();
  for (int p=0; p < npairs; p++) {
    const Objpair& objpair = pt.get_pair (p);
    int i = objpair.i();
    int j = objpair.j();
    double igrad[3], jgrad[3];
    bool badflag = false;
    double dot = dot_and_gradient (ev.obj (i).p,
                                   ev.obj (j).p,
                                   use_e_flag,
                                   igrad,
                                   jgrad,
                                   badflag);
    if (badflag)
      return false;

    for (std::vector<Constraint>::size_type k=0; k < constraints.size(); k++)
      if (objpair.for_constraint (k)) {
    F(k+1) += objpair.for_constraint (k) * dot;
    addin_gradients (ev, k+1, objpair.for_constraint (k),
             i, igrad, j, jgrad, Bx, By);
      }

  }

  add_mass_terms (ev, constraints, F);
  return true;
}


/**
    @brief Helper function: Perform the Jacobian tranformation for
    \f$(p_{\nu}^{x},p_{\nu}^{y}) \to (k_{T}^{x},k_{T}^{y})\f$ for
    a given object.

    @param ndx Index of the object for which to transform gradients.

    @param v The object's four-momentum.

    @param Bx The gradients of the well-measured variables.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param kt_ndx The index of the \f$k_{T}\f$ variables in the variables
    array.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.
 */
void add_nuterm (unsigned ndx,
                 const Fourvec& v,
                 Matrix& Bx,
                 bool use_e_flag,
                 int kt_ndx)
//
// Purpose: Carry out the Jacobian transformation for
//          (p_nu^x,p_nu_y) -> (kt^x, kt_y) for a given object.
//
// Inputs:
//   ndx -         Index of the object for which to transform gradients.
//   v -           The object's 4-momentum.
//   Bx -          The well-measured variable gradients.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//   kt_ndx -      The index of the kt variables in the variables array.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//
{
  double px = v.px();
  double py = v.py();
  double cot_theta = v.pz() / v.vect().perp();

  for (int j=1; j<=Bx.num_col(); j++) {
    double dxnu = Bx(kt_ndx+x_offs, j);
    double dynu = Bx(kt_ndx+y_offs, j);

    if (dxnu != 0 || dynu != 0) {
      double fac = 1 / v.vect().mag();
      if (use_e_flag)
    fac = v.e() * fac * fac;
      Bx(ndx +   p_offs, j) -= (px*dxnu + py*dynu) * fac;
      Bx(ndx + phi_offs, j) +=  py*dxnu - px*dynu;
      Bx(ndx + eta_offs, j) -= (px*dxnu + py*dynu) * cot_theta;
    }
  }
}


/**
    @brief Helper function: Carry out the Jacobian transformations for
    the neutrino.  First, convert from spherical coordinates
    \f$(p,\phi,\eta)\f$ to rectangular \f$(x,y,z)\f$.  Then convert from
    neutrino \f$p_{T}\f$ components to \f$k_{T}\f$ components.

    @param ev The event we are fitting.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param Bx The gradients of the well-measured variables, should be passed
    with the correct dimensionaly.

    @param By The gradients of the poorly-measured variables, should be
    passed with the scorrect dimensionaly.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.
    - <i>By</i> The updated gradients of the poorly-measured variables.
 */
void convert_neutrino (const Fourvec_Event& ev,
                       bool use_e_flag,
                       Matrix& Bx,
                       Matrix& By)
//
// Purpose: Carry out the Jacobian transformations the neutrino.
//          First, convert from spherical (p, phi, theta) coordinates
//          to rectangular (x, y, z).  Then convert from neutrino pt
//          components to kt components.
//
// Inputs:
//   ev -          The event we're fitting.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//   By -          The poorly-measured variable gradients, updated.
//
{
  int nconstraints = Bx.num_col ();

  const Fourvec& nu = ev.nu ();

  // convert neutrino from polar coordinates to rectangular.
  double pnu2  = nu.vect().mag2();  double pnu = sqrt (pnu2);
  double ptnu2 = nu.vect().perp2(); double ptnu = sqrt (ptnu2);

  // Doesn't matter whether we use E or P here, since nu is massless.

  double thfac = nu.z()/pnu2/ptnu;
  double fac[3][3];
  fac[0][0] = nu(0)/pnu;     fac[0][1] = nu(1)/pnu;     fac[0][2] = nu(2)/pnu;
  fac[1][0] = - nu(1)/ptnu2; fac[1][1] =   nu(0)/ptnu2; fac[1][2] = 0;
  fac[2][0] = nu(0)*thfac;   fac[2][1] = nu(1)*thfac;   fac[2][2] = -ptnu/pnu2;

  int kt_ndx = obj_index (ev.nobjs());
  for (int j=1; j<=nconstraints; j++) {
    double tmp1 = fac[0][0]*Bx(kt_ndx+x_offs,j) +
                  fac[1][0]*Bx(kt_ndx+y_offs,j) +
                  fac[2][0]*By(nu_z,j);
    Bx(kt_ndx+y_offs,j) = fac[0][1]*Bx(kt_ndx+x_offs,j) +
                          fac[1][1]*Bx(kt_ndx+y_offs,j) +
                          fac[2][1]*By(nu_z,j);
    By(nu_z,j) = fac[0][2]*Bx(kt_ndx+x_offs,j) + fac[2][2]*By(nu_z,j);

    Bx(kt_ndx+x_offs,j) = tmp1;
  }

  // Add nu terms.
  for (int j=0; j<ev.nobjs(); j++) {
    add_nuterm (obj_index (j), ev.obj(j).p, Bx, use_e_flag, kt_ndx);
  }
}


/**
    @brief Helpere function: Calculate the overall \f$k_{T}\f$ constraints
    \f$(k_{T} = 0)\f$ for the case where there is no neutrino.

    @param ev The event we are fitting.

    @param use_e_flag If TRUE, then we are using energy <i>E</i> as the
    fit variable.<br>
    If FALSE, we are using magnitude of three-momentum <i>p</i> as the
    fit variable.

    @param F Vector of constraint values.

    @param Bx The gradients of the well-measured variables, should be passed
    with the correct dimensionaly.

    @par Output:
    - <i>F</i> The updated vectgor of constraints values.<br>
    - <i>Bx</i> The updated gradients of the well-measured variables.
 */
void calculate_kt_constraints (const Fourvec_Event& ev,
                               bool use_e_flag,
                               Row_Vector& F,
                               Matrix& Bx)
//
// Purpose: Calculate the overall kt constraints (kt = 0) for the case
//          where there is no neutrino.
//
// Inputs:
//   ev -          The event we're fitting.
//   use_e_flag -  If true, we're using E as the fit variable, otherwise p.
//   F -           Vector of constraint values.
//   Bx -          The well-measured variable gradients.
//
// Outputs:
//   F -           Vector of constraint values, updated.
//   Bx -          The well-measured variable gradients, updated.
//
{
  Fourvec tmp = Fourvec(0);
  int base = F.num_col() - 2;
  int nobjs = ev.nobjs();
  for (int j=0; j<nobjs; j++) {
    const Fourvec& obj = ev.obj (j).p;
    tmp += obj;

    int ndx = obj_index (j);
    double p = obj.vect().mag();
    double cot_theta = obj.z() / obj.vect().perp();

    Bx(ndx +   p_offs, base+1) = obj(0) / p;
    Bx(ndx + phi_offs, base+1) = -obj(1);
    Bx(ndx + eta_offs, base+1) = obj(0) * cot_theta;

    Bx(ndx +   p_offs, base+2) = obj(1) / p;
    Bx(ndx + phi_offs, base+2) = obj(0);
    Bx(ndx + eta_offs, base+2) = obj(1) * cot_theta;

    if (use_e_flag) {
      Bx(ndx +   p_offs, base+1) *= obj.e() / p;
      Bx(ndx +   p_offs, base+2) *= obj.e() / p;
    }
  }

  int kt_ndx = obj_index (nobjs);
  Bx(kt_ndx+x_offs, base+1) = -1;
  Bx(kt_ndx+y_offs, base+2) = -1;

  F(base+1) = tmp(0) - ev.kt().x ();
  F(base+2) = tmp(1) - ev.kt().y ();
}


/**
    @brief Do the Jacobian transformation from \f$\theta\f$ to \f$\eta\f$
    for a single object.

    @param i The index of the object.

    @param cos_theta The \f$\cos \theta\f$ for the object.

    @param Bx The gradients of the well-measured variables.

    @par Output:
    - <i>Bx</i> The updated gradients of the well-measured variables.
 */
void ddtheta_to_ddeta (int i, double cos_theta, Matrix& Bx)
//
// Purpose: Do the Jacobian transformation from theta -> eta
//          for a single object.
//
// Inputs:
//   i -           The index of the object.
//   cos_theta -   cos(theta) for the object.
//   Bx -          The well-measured variable gradients.
//
// Outputs:
//   Bx -          The well-measured variable gradients, updated.
//
{
  double sin_theta = sqrt (1 - cos_theta * cos_theta);
  for (int j=1; j<=Bx.num_col(); j++)
    Bx(i,j) *= - sin_theta;   /* \sin\theta = 1 / \cosh\eta */
}


} // unnamed namespace


/**
    @brief Concrete realization of the Constraint_Calculator class.
    Evaluate constraints at the point described by <i>x</i> and
    <i>y</i>
    (well-measured and poorly-measured variables, respectively).  The results
    should be stored in <i>F</i>.  <i>Bx</i> and <i>By</i> should be set to
    the gradients of <i>F</i> with respect to <i>x</i> and <i>y</i>,
    respectively.

    @param x  Column_Vector of well-measured variables.

    @param y  Column_Vector of poorly-measured variables.

    @param F  Row_Vector contains the results of the constraint evaluation.

    @param Bx Gradients of <i>F</i> with respect to <i>x</i>

    @param By Gradients of <i>F</i> with respect to <i>y</i>

    @par Return:
    <b>true</b> if the point <i>(x,y)</i> is accepted.<br>
    <b>false</b> if the point <i>(x,y)</i> is rejected
    (i.e., in an unphysical region).  The constraints need not be
    evaluated in that case.
 */
class Fourvec_Constraint_Calculator
  : public Constraint_Calculator
//
// Purpose: Constraint evaluator.
//
{
public:
  // Constructor, destructor.
  Fourvec_Constraint_Calculator (Fourvec_Event& ev,
                                 const vector<Constraint>& constraints,
                                 const Fourvec_Constrainer_Args& args);
  virtual ~Fourvec_Constraint_Calculator () {}

  // Evaluate constraints at the point described by X and Y (well-measured
  // and poorly-measured variables, respectively).  The results should
  // be stored in F.  BX and BY should be set to the gradients of F with
  // respect to X and Y, respectively.
  //
  // Return true if the point X, Y is accepted.
  // Return false if it is rejected (i.e., in an unphysical region).
  // The constraints need not be evaluated in that case.
  virtual bool eval (const Column_Vector& x,
                     const Column_Vector& y,
                     Row_Vector& F,
                     Matrix& Bx,
                     Matrix& By);


  // Calculate the constraint functions and gradients.
  bool calculate_constraints (Row_Vector& F,
                              Matrix& Bx,
                              Matrix& By) const;

private:
  // The event we're fitting.
  Fourvec_Event& _ev;

  // Vector of constraints.
  const vector<Constraint>& _constraints;

  // Argument values.
  const Fourvec_Constrainer_Args& _args;

  // The pair table.
  Pair_Table _pt;
};


/**
    @brief Constructor

    @param ev The event we are fitting.

    @param constraints The list of constraints.

    @param args The parameter settings for this instance.
 */
Fourvec_Constraint_Calculator::Fourvec_Constraint_Calculator
  (Fourvec_Event& ev,
   const vector<Constraint>& constraints,
   const Fourvec_Constrainer_Args& args)
//
// Purpose: Constructor.
//
// Inputs:
//   ev -          The event we're fitting.
//   constraints - The list of constraints.
//   args -        The parameter settings.
//
//
  : Constraint_Calculator (constraints.size() +
                           ((ev.has_neutrino() || args.ignore_met()) ? 0 : 2)),
    _ev (ev),
    _constraints (constraints),
    _args (args),
    _pt (constraints, ev)

{
}


/**
    @brief Calculate the constraint functions and gradients.

    @param F Vector of constraint values.

    @param Bx The gradient of well-measured variables.

    @param By The gradient of poorly-measured variables.

    @par Output:
    - <i>F</i>.
    - <i>Bx</i>.
    - <i>By</i>.
 */
bool
Fourvec_Constraint_Calculator::calculate_constraints (Row_Vector& F,
                                                      Matrix& Bx,
                                                      Matrix& By) const
//
// Purpose: Calculate the constraint functions and gradients.
//
// Outputs:
//   F -           Vector of constraint values.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
{
  // Clear the matrices.
  Bx = Matrix (Bx.num_row(), Bx.num_col(), 0);
  By = Matrix (By.num_row(), By.num_col(), 0);
  F = Row_Vector (F.num_col(), 0);

  const double p_eps = 1e-10;

  if (_ev.has_neutrino() && _ev.nu().z() > _args.e_com()) {
    return false;
  }

  int nobjs = _ev.nobjs ();

  // Reject the point if any of the momenta get too small.
  for (int j=0; j<nobjs; j++) {
    if (_ev.obj(j).p.perp() <= p_eps || _ev.obj(j).p.e() <= p_eps) {
      return false;
    }
  }

  if (! calculate_mass_constraints (_ev, _pt, _constraints, _args.use_e(),
                                    F, Bx, By))
    return false;

  if (_ev.has_neutrino())
    convert_neutrino (_ev, _args.use_e(), Bx, By);
  else if (!_args.ignore_met())
  {
    /* kt constraints */
    calculate_kt_constraints (_ev, _args.use_e(), F, Bx);
  }

  /* convert d/dtheta to d/deta */
  for (int j=0; j<nobjs; j++) {
    ddtheta_to_ddeta (obj_index (j) + eta_offs,
                      _ev.obj(j).p.cosTheta(),
                      Bx);
  }

  /* handle muons */
  for (int j=0; j<nobjs; j++) {
    const FE_Obj& obj = _ev.obj (j);
    if (obj.muon_p) {
      // Again, E vs. P doesn't matter here since we assume muons to be massless.
      double pmu2 = obj.p.vect().mag2();
      int ndx = obj_index (j) + p_offs;
      for (int k=1; k<=Bx.num_col(); k++)
    Bx(ndx, k) = - Bx(ndx, k) * pmu2;
    }
  }

  return true;
}


/**
    @brief Evaluate constraints at the point described by <i>x</i> and
    <i>y</i>
    (well-measured and poorly-measured variables, respectively).  The results
    should be stored in <i>F</i>.  <i>Bx</i> and <i>By</i> should be set to
    the gradients of <i>F</i> with respect to <i>x</i> and <i>y</i>,
    respectively.

    @param x  Column_Vector of well-measured variables.

    @param y  Column_Vector of poorly-measured variables.

    @param F  Row_Vector contains the results of the constraint evaluation.

    @param Bx Gradients of <i>F</i> with respect to <i>x</i>

    @param By Gradients of <i>F</i> with respect to <i>y</i>

    @par Output:
    - <i>F</i>.
    - <i>Bx</i>.
    - <i>By</i>.

    @par Return:
    <b>true</b> if the point <i>(x,y)</i> is accepted.<br>
    <b>false</b> if the point <i>(x,y)</i> is rejected
    (i.e., in an unphysical region).  The constraints need not be
    evaluated in that case.
 */
bool Fourvec_Constraint_Calculator::eval (const Column_Vector& x,
                                          const Column_Vector& y,
                                          Row_Vector& F,
                                          Matrix& Bx,
                                          Matrix& By)
//
// Purpose: Evaluate constraints at the point described by X and Y
//          (well-measured and poorly-measured variables, respectively).
//          The results should be stored in F.  BX and BY should be set
//          to the gradients of F with respect to X and Y, respectively.
//
// Inputs:
//   x -           Vector of well-measured variables.
//   y -           Vector of poorly-measured variables.
//
// Outputs:
//   F -           Vector of constraint values.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
{
  int nobjs = _ev.nobjs();

  const double p_eps = 1e-10;
  const double eta_max = 10;

  // Give up if we've gone into an obviously unphysical region.
  for (int j=0; j<nobjs; j++)
    if (x(obj_index (j) + p_offs) < p_eps ||
        fabs(x(obj_index (j) + eta_offs)) > eta_max) {
      return false;
    }

  unpack_event (_ev, x, y, _args.use_e(),
                _ev.has_neutrino() || !_args.ignore_met());

  return calculate_constraints (F, Bx, By);
}


//*************************************************************************
// Mass calculation.
//


namespace {


/**
    @brief Helper function: Calculate the error propagation to find the
    uncertainty in the final mass.

    @param Q The final error matrix for the well-measured variables.

    @param R The final error matrix for the poorly-measured variables.

    @param S The final cross error matrix for the two sets of variables.

    @param Bx The gradient of the well-measured variables.

    @param By The gradient of the poorly-measured variables.

    @par Return:
    The uncertainty in the final mass.
 */
double calculate_sigm (const Matrix& Q,
                       const Matrix& R,
                       const Matrix& S,
                       const Column_Vector& Bx,
                       const Column_Vector& By)
//
// Purpose: Do error propagation to find the uncertainty in the final mass.
//
// Inputs:
//   Q -           The final error matrix for the well-measured variables.
//   R -           The final error matrix for the poorly-measured variables.
//   S -           The final cross error matrix for the two sets of variables.
//   Bx -          The well-measured variable gradients.
//   By -          The poorly-measured variable gradients.
//
{
  double sig2 = scalar (Bx.T() * Q * Bx);

  if (By.num_row() > 0) {
    sig2 += scalar (By.T() * R * By);
    sig2 += 2 * scalar (Bx.T() * S * By);
  }

  assert (sig2 >= 0);
  return sqrt (sig2);
}


/**
    @brief Helper function: Calculate the final requested mass and
    its uncertainty.

    @param ev The event we are fitting.

    @param mass_constraint The description of the mass to be calculated.

    @param args Parameter settings.

    @param chisq The \f$\chi^{2}\f$ from the fit.

    @param Q The final error matrix for the well-measured variables.

    @param R The final error matrix for the poorly-measured variables.

    @param S The final cross error matrix for the two sets of variables.

    @param m The mass (output).

    @param sigm The uncertainty on the mass (output).
 */
void calculate_mass (Fourvec_Event& ev,
                     const vector<Constraint>& mass_constraint,
                     const Fourvec_Constrainer_Args& args,
                     double chisq,
                     const Matrix& Q,
                     const Matrix& R,
                     const Matrix& S,
                     double& m,
                     double& sigm)
//
// Purpose: Calculate the final requested mass and its uncertainty.
//
// Inputs:
//   ev -          The event we're fitting.
//   mass_constraint- The description of the mass we're to calculate.
//   args -        Parameter settings.
//   chisq -       The chisq from the fit.
//   Q -           The final error matrix for the well-measured variables.
//   R -           The final error matrix for the poorly-measured variables.
//   S -           The final cross error matrix for the two sets of variables.
//
// Outputs:
//   m -           The mass.
//   sigm -        Its uncertainty.
//
{
  // Don't do anything if the mass wasn't specified.
  if (mass_constraint.size () == 0) {
    m = 0;
    sigm = 0;
    return;
  }

  // Do the constraint calculation.
  int n_measured_vars = ev.nobjs()*3 + 2;
  int n_unmeasured_vars = 0;
  if (ev.has_neutrino ()) n_unmeasured_vars = 1;

  Row_Vector F(1);
  Matrix Bx (n_measured_vars, 1);
  Matrix By (n_unmeasured_vars, 1);

  Fourvec_Constraint_Calculator cc (ev, mass_constraint, args);
  cc.calculate_constraints (F, Bx, By);

  // Calculate the mass.
  //assert (F(1) >= 0);
  if (F(1) >= 0.)
    m = sqrt (F(1) * 2);
  else {
    m = 0.;
    chisq = -100.;
  }

  // And the uncertainty.
  // We can only do this if the fit converged.
  if (chisq < 0)
    sigm = 0;
  else {
    //assert (F(1) > 0);
    Bx = Bx / (m);
    By = By / (m);

    sigm = calculate_sigm (Q, R, S, Bx, By);
  }
}


} // unnamed namespace


double Fourvec_Constrainer::constrain (Fourvec_Event& ev,
                                       double& m,
                                       double& sigm,
                                       Column_Vector& pullx,
                                       Column_Vector& pully)
//
// Purpose: Do a constrained fit for EV.  Returns the requested mass and
//          its error in M and SIGM, and the pull quantities in PULLX and
//          PULLY.  Returns the chisq; this will be < 0 if the fit failed
//          to converge.
//
// Inputs:
//   ev -          The event we're fitting.
//
// Outputs:
//   ev -          The fitted event.
//   m -           Requested invariant mass.
//   sigm -        Uncertainty on m.
//   pullx -       Pull quantities for well-measured variables.
//   pully -       Pull quantities for poorly-measured variables.
//
// Returns:
//   The fit chisq, or < 0 if the fit didn't converge.
//
{
  adjust_fourvecs (ev, _args.use_e ());

  bool use_kt = ev.has_neutrino() || !_args.ignore_met();
  int nobjs = ev.nobjs ();
  int n_measured_vars = nobjs * 3;
  int n_unmeasured_vars = 0;
  int n_constraints = _constraints.size ();

  if (use_kt) {
    n_measured_vars += 2;

    if (ev.has_neutrino ())
      n_unmeasured_vars = 1;
    else
      n_constraints += 2;
  }

  Matrix G_i (n_measured_vars, n_measured_vars);
  Diagonal_Matrix Y (n_unmeasured_vars);
  Column_Vector x (n_measured_vars);
  Column_Vector y (n_unmeasured_vars);
  pack_event (ev, _args.use_e(), use_kt, x, y, G_i, Y);

  Column_Vector xm = x;
  Column_Vector ym = y;

  // ??? Should allow for using a different underlying fitter here.
  Chisq_Constrainer fitter (_args.chisq_constrainer_args());

  Fourvec_Constraint_Calculator cc (ev, _constraints, _args);

  Matrix Q;
  Matrix R;
  Matrix S;
  double chisq = fitter.fit (cc,
                             xm, x, ym, y, G_i, Y,
                             pullx, pully,
                             Q, R, S);

  unpack_event (ev, x, y, _args.use_e (), use_kt);

  calculate_mass (ev, _mass_constraint, _args, chisq, Q, R, S, m, sigm);

  return chisq;
}


} // namespace hitfit
