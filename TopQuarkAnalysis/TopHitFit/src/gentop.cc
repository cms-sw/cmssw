//
// $Id: gentop.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/gentop.cc
// Purpose: Toy ttbar event generator for testing.
// Created: Jul, 2000, sss.
//
// CMSSW File      : src/gentop.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file gentop.cc

    @brief A toy event generator for
    \f$t\bar{t} \to \ell + 4~\mathrm{jets}\f$ events. This file also contains
    helper function for generation of random toy events.
    See the documentation for the header file gentop.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/gentop.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandBreitWigner.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <cmath>
#include <ostream>


using std::ostream;


// Event number counter.
/**
    @brief Event number counter.
 */
namespace {
int next_evnum = 0;
}


namespace hitfit {


//**************************************************************************
// Argument handling.
//


Gentop_Args::Gentop_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _t_pt_mean      (defs.get_float ("t_pt_mean")),
    _mt             (defs.get_float ("mt")),
    _sigma_mt       (defs.get_float ("sigma_mt")),
    _mh             (defs.get_float ("mh")),
    _sigma_mh       (defs.get_float ("sigma_mh")),
    _recoil_pt_mean (defs.get_float ("recoil_pt_mean")),
    _boost_sigma    (defs.get_float ("boost_sigma")),
    _m_boost        (defs.get_float ("m_boost")),
    _mb             (defs.get_float ("mb")),
    _sigma_mb       (defs.get_float ("sigma_mb")),
    _mw             (defs.get_float ("mw")),
    _sigma_mw       (defs.get_float ("sigma_mw")),
    _svx_tageff     (defs.get_float ("svx_tageff")),
    _smear          (defs.get_bool ("smear")),
    _smear_dir      (defs.get_bool ("smear_dir")),
    _muon           (defs.get_bool ("muon")),
    _ele_res_str    (defs.get_string ("ele_res_str")),
    _muo_res_str    (defs.get_string ("muo_res_str")),
    _jet_res_str    (defs.get_string ("jet_res_str")),
    _kt_res_str     (defs.get_string ("kt_res_str"))
{
}


double Gentop_Args::t_pt_mean () const
//
// Purpose: Return the t_pt_mean parameter.
//          See the header for documentation.
//
{
  return _t_pt_mean;
}


double Gentop_Args::mt () const
//
// Purpose: Return the mt parameter.
//          See the header for documentation.
//
{
  return _mt;
}


double Gentop_Args::sigma_mt () const
//
// Purpose: Return the sigma_mt parameter.
//          See the header for documentation.
//
{
  return _sigma_mt;
}


double Gentop_Args::svx_tageff () const
//
// Purpose: Return the svx_tageff parameter.
//          See the header for documentation.
//
{
  return _svx_tageff;
}


double Gentop_Args::mh () const
//
// Purpose: Return the mh parameter.
//          See the header for documentation.
//
{
  return _mh;
}


double Gentop_Args::sigma_mh () const
//
// Purpose: Return the sigma_mh parameter.
//          See the header for documentation.
//
{
  return _sigma_mh;
}


bool Gentop_Args::smear () const
//
// Purpose: Return the smear parameter.
//          See the header for documentation.
//
{
  return _smear;
}


bool Gentop_Args::smear_dir () const
//
// Purpose: Return the smear_dir parameter.
//          See the header for documentation.
//
{
  return _smear_dir;
}


bool Gentop_Args::muon () const
//
// Purpose: Return the muon parameter.
//          See the header for documentation.
//
{
  return _muon;
}


double Gentop_Args::recoil_pt_mean () const
//
// Purpose: Return the recoil_pt_mean parameter.
//          See the header for documentation.
//
{
  return _recoil_pt_mean;
}


double Gentop_Args::boost_sigma () const
//
// Purpose: Return the boost_sigma parameter.
//          See the header for documentation.
//
{
  return _boost_sigma;
}


double Gentop_Args::m_boost () const
//
// Purpose: Return the m_boost parameter.
//          See the header for documentation.
//
{
  return _m_boost;
}


double Gentop_Args::mb () const
//
// Purpose: Return the mb parameter.
//          See the header for documentation.
//
{
  return _mb;
}


double Gentop_Args::sigma_mb () const
//
// Purpose: Return the sigma_mb parameter.
//          See the header for documentation.
//
{
  return _sigma_mb;
}


double Gentop_Args::mw () const
//
// Purpose: Return the mw parameter.
//          See the header for documentation.
//
{
  return _mw;
}


double Gentop_Args::sigma_mw () const
//
// Purpose: Return the sigma_mw parameter.
//          See the header for documentation.
//
{
  return _sigma_mw;
}


std::string Gentop_Args::ele_res_str () const
//
// Purpose: Return the ele_res_str parameter.
//          See the header for documentation.
//
{
  return _ele_res_str;
}


std::string Gentop_Args::muo_res_str () const
//
// Purpose: Return the muo_res_str parameter.
//          See the header for documentation.
//
{
  return _muo_res_str;
}


std::string Gentop_Args::jet_res_str () const
//
// Purpose: Return the jet_res_str parameter.
//          See the header for documentation.
//
{
  return _jet_res_str;
}


std::string Gentop_Args::kt_res_str () const
//
// Purpose: Return the kt_res_str parameter.
//          See the header for documentation.
//
{
  return _kt_res_str;
}


//**************************************************************************
// Internal helper functions.
//


namespace {


/**
    @brief Generate a unit vector with \f$(\theta,\phi)\f$ uniformly
    distributed over the sphere.

    @param engine The underlying random number generator.
 */
Threevec rand_spher (CLHEP::HepRandomEngine& engine)
//
// Purpose: Return a unit vector with (theta, phi) uniformly distributed
//          over a sphere.
//
// Inputs:
//   engine -      The underlying RNG.
//
// Returns:
//   The generated vector.
//
{
  CLHEP::RandFlat r (engine);

  Threevec v;

  double U = r.fire(0.0,1.0);
  double V = r.fire(0.0,1.0);

  double theta = 2.0*CLHEP::pi*U ;
  double phi   = acos(2*V - 1.0);

  double x = sin(theta)*cos(phi);
  double y = sin(theta)*sin(phi);
  double z = cos(theta);

  v = Threevec(x,y,z);

  return v.unit ();
}


/**
    @brief Given a direction, mass, and width, chose a mass from a
    Breit-Wigner distribution and return a four-momentum with the chosen
    mass and the specified direction.

    @param p The direction three-momenta
    (not necessary to have unit magnitude).

    @param m_true The mean for the Breit-Wigner distribution.

    @param sigma The width for the Breit-Wigner distribution.

    @param engine The underlying random number generator.
 */
Fourvec make_massive (const Threevec& p,
                      double m_true,
                      double sigma,
                      CLHEP::HepRandomEngine& engine)
//
// Purpose: Given a direction, mass, and width, choose a mass from a
//          Breit-Wigner and return a 4-vector with the chosen mass
//          and the specified direction.
//
// Inputs:
//   p -           The direction.
//   m_true -      The mean for the Breit-Wigner.
//   sigma -       The width for the Breit-Wigner.
//   engine -      The underlying RNG.
//
// Returns:
//   The generated 4-vector.
//
{
  CLHEP::RandBreitWigner rbw (engine);
  double m = rbw.fire (m_true, sigma);
  return Fourvec (p, sqrt (m*m + p.mag2()));
}

/**
    @brief Decay a particle with initial four-momentum \f$v\f$ into
    two particles with mass \f$m_{1}\f$ and \f$m_{2}\f$.

    @param v The initial four-momentum.

    @param m1 Mass of the first decay product.

    @param m2 Mass of the second decay product.

    @param engine The underlying random number generator.

    @param vout1 Output, the outgoing four-momentum of the first decay product.

    @param vout2 Output, the outgoing four-momentum of the second decay
    product.
 */
void decay (const Fourvec& v, double m1, double m2,
            CLHEP::HepRandomEngine& engine,
            Fourvec& vout1, Fourvec& vout2)
//
// Purpose: v decays into two particles w/masses m1, m2.
//
// Inputs:
//   v -           The incoming 4-vector.
//   m1 -          Mass of the first decay product.
//   m2 -          Mass of the second decay product.
//   engine -      The underlying RNG.
//
// Outputs:
//   vout1 -       Outgoing 4-vector of the first decay product.
//   vout2 -       Outgoing 4-vector of the second decay product.
//
{
  // Construct a decay in the incoming particle's rest frame,
  // uniformly distributed in direction.
  Threevec p = rand_spher (engine);
  double m0 = v.m();

  if (m1 + m2 > m0) {
    // What ya gonna do?
    double f = m0 / (m1 + m2);
    m1 *= f;
    m2 *= f;
  }

  double m0_2 = m0*m0;
  double m1_2 = m1*m1;
  double m2_2 = m2*m2;

  // Calculate the 3-momentum of each particle in the decay frame.
  p *= 0.5/m0 * sqrt (    m0_2*m0_2 +   m1_2*m1_2 +   m2_2*m2_2
                      - 2*m0_2*m1_2 - 2*m0_2*m2_2 - 2*m1_2*m2_2);
  double p2 = p.mag2();

  vout1 = Fourvec ( p, sqrt (p2 + m1_2));
  vout2 = Fourvec (-p, sqrt (p2 + m2_2));

  // Boost out of the rest frame.
  vout1.boost (v.boostVector ());
  vout2.boost (v.boostVector ());
}


/**
    @brief Generate a vector in a spherically uniform random direction,
    with transverse momentum \f$p_{T}\f$ drawn from an exponential distribution
    with mean <i>pt_mean</i>.

    @param pt_mean The mean of the distribution.

    @param engine The underlying random number generator.
 */
Threevec rand_pt (double pt_mean,
                  CLHEP::HepRandomEngine& engine)
//
// Purpose: Generate a vector in a (uniformly) random direction,
//          with pt chosen from an exponential distribution with mean pt_mean.
//
// Inputs:
//   pt_mean -     The mean of the distribution.
//   engine -      The underlying RNG.
//
// Returns:
//   The generated vector.
//
{
  CLHEP::RandExponential rexp (engine);

  // A random direction.
  Threevec p = rand_spher (engine);

  // Scale by random pt.
  p *= (rexp.fire (pt_mean) / p.perp());

  return p;
}


/**
    @brief Generate a random boost for the event.

    @param args The parameter settings.

    @param engine The underlying random number generator.
 */
Fourvec rand_boost (const Gentop_Args& args, CLHEP::HepRandomEngine& engine)
//
// Purpose: Generate a random boost for the event.
//
// Inputs:
//   args -        The parameter settings.
//   engine -      The underlying RNG.
//
// Returns:
//   The generated boost.
//
{
  CLHEP::RandExponential rexp (engine);
  CLHEP::RandFlat rflat (engine);
  CLHEP::RandGauss rgauss (engine);

  // Boost in pt and z.
  Threevec p (1, 0, 0);
  p.rotateZ (rflat.fire (0, 2 * M_PI));
  p *= rexp.fire (args.recoil_pt_mean());
  p.setZ (rgauss.fire (0, args.boost_sigma()));
  return Fourvec (p, sqrt (p.mag2() + args.m_boost()*args.m_boost()));
}


/**
    @brief Simulate SVX (Secondary Vertex) tagging.

    @param args The parameter settings.

    @param ev The event to tag.

    @param engine The underlying random number generator.
 */
void tagsim (const Gentop_Args& args,
             Lepjets_Event& ev,
             CLHEP::HepRandomEngine& engine)
//
// Purpose: Simulate SVX tagging.
//
// Inputs:
//   args -        The parameter settings.
//   ev -          The event to tag.
//   engine -      The underlying RNG.
//
// Outputs:
//   ev -          The event with tags filled in.
//
{
  CLHEP::RandFlat rflat (engine);
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < ev.njets(); i++) {
    int typ = ev.jet(i).type();
    if (typ == hadb_label || typ == lepb_label || typ == higgs_label) {
      if (rflat.fire() < args.svx_tageff())
        ev.jet(i).svx_tag() = true;
    }
  }
}


} // unnamed namespace


//**************************************************************************
// External interface.
//


Lepjets_Event gentop (const Gentop_Args& args,
                      CLHEP::HepRandomEngine& engine)
//
// Purpose: Generate a ttbar -> ljets event.
//
// Inputs:
//   args -        The parameter settings.
//   engine -      The underlying RNG.
//
// Returns:
//   The generated event.
//
{
  CLHEP::RandBreitWigner rbw (engine);
  CLHEP::RandGauss rgauss (engine);

  // Get the t decay momentum in the ttbar rest frame.
  Threevec p = rand_pt (args.t_pt_mean(), engine);

  // Make the t/tbar vectors.
  Fourvec lept = make_massive ( p, args.mt(), args.sigma_mt(), engine);
  Fourvec hadt = make_massive (-p, args.mt(), args.sigma_mt(), engine);

  // Boost the rest frame.
  Fourvec boost = rand_boost (args, engine);
  lept.boost (boost.boostVector());
  hadt.boost (boost.boostVector());

  // Decay t -> b W, leptonic side.
  Fourvec lepb, lepw;
  double mlb = rgauss.fire (args.mb(), args.sigma_mb());
  double mlw = rbw.fire (args.mw(), args.sigma_mw());
  decay (lept, 
         mlb,
         mlw,
         engine,
         lepb,
         lepw);

  // Decay t -> b W, hadronic side.
  Fourvec hadb, hadw;
  double mhb = rgauss.fire (args.mb(), args.sigma_mb());
  double mhw = rbw.fire (args.mw(), args.sigma_mw());
  decay (hadt, 
         mhb,
         mhw,
         engine,
         hadb,
         hadw);

  // Decay W -> l nu.
  Fourvec lep, nu;
  decay (lepw, 0, 0, engine, lep, nu);

  // Decay W -> qqbar.
  Fourvec q1, q2;
  decay (hadw, 0, 0, engine, q1, q2);

  // Fill in the event.
  Lepjets_Event ev (0, ++next_evnum);
  Vector_Resolution lep_res (args.muon() ? 
                             args.muo_res_str() :
                             args.ele_res_str());
  Vector_Resolution jet_res (args.jet_res_str());
  Resolution kt_res = (args.kt_res_str());

  ev.add_lep (Lepjets_Event_Lep (lep,
                                 args.muon() ? muon_label : electron_label,
                                 lep_res));

  ev.add_jet (Lepjets_Event_Jet (lepb,  lepb_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (hadb,  hadb_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (  q1, hadw1_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (  q2, hadw2_label, jet_res));

  ev.met() = nu;
  ev.kt_res() = kt_res;

  // Simulate SVX tagging.
  tagsim (args, ev, engine);

  // Smear the event, if requested.
  if (args.smear())
    ev.smear (engine, args.smear_dir());

  // Done!
  return ev;
}


Lepjets_Event gentth (const Gentop_Args& args,
                      CLHEP::HepRandomEngine& engine)
//
// Purpose: Generate a ttH -> ljets event.
//
// Inputs:
//   args -        The parameter settings.
//   engine -      The underlying RNG.
//
// Returns:
//   The generated event.
//
{
  CLHEP::RandBreitWigner rbw (engine);
  CLHEP::RandGauss rgauss (engine);

  // Generate three-vectors for two tops.
  Threevec p_t1 = rand_pt (args.t_pt_mean(), engine);
  Threevec p_t2 = rand_pt (args.t_pt_mean(), engine);

  // Conserve momentum.
  Threevec p_h = -(p_t1 + p_t2);

  // Construct the 4-vectors.
  Fourvec lept = make_massive  (p_t1, args.mt(), args.sigma_mt(), engine);
  Fourvec hadt = make_massive  (p_t2, args.mt(), args.sigma_mt(), engine);
  Fourvec higgs = make_massive ( p_h, args.mh(), args.sigma_mh(), engine);

  // Boost the rest frame.
  Fourvec boost = rand_boost (args, engine);
  lept.boost (boost.boostVector());
  hadt.boost (boost.boostVector());
  higgs.boost (boost.boostVector());

  // Decay t -> b W, leptonic side.
  Fourvec lepb, lepw;
  decay (lept, 
         rgauss.fire (args.mb(), args.sigma_mb()),
         rbw.fire (args.mw(), args.sigma_mw()),
         engine,
         lepb,
         lepw);

  // Decay t -> b W, hadronic side.
  Fourvec hadb, hadw;
  decay (hadt, 
         rgauss.fire (args.mb(), args.sigma_mb()),
         rbw.fire (args.mw(), args.sigma_mw()),
         engine,
         hadb,
         hadw);

  // Decay W -> l nu.
  Fourvec lep, nu;
  decay (lepw, 0, 0, engine, lep, nu);

  // Decay W -> qqbar.
  Fourvec q1, q2;
  decay (hadw, 0, 0, engine, q1, q2);

  // Decay H -> bbbar.
  Fourvec hb1, hb2;
  decay (higgs, 
         rgauss.fire (args.mb(), args.sigma_mb()),
         rgauss.fire (args.mb(), args.sigma_mb()),
         engine,
         hb1,
         hb2);

  // Fill in the event.
  Lepjets_Event ev (0, ++next_evnum);
  Vector_Resolution lep_res (args.muon() ? 
                             args.muo_res_str() :
                             args.ele_res_str());
  Vector_Resolution jet_res (args.jet_res_str());
  Resolution kt_res = (args.kt_res_str());

  ev.add_lep (Lepjets_Event_Lep (lep,
                                 args.muon() ? muon_label : electron_label,
                                 lep_res));

  ev.add_jet (Lepjets_Event_Jet (lepb,  lepb_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (hadb,  hadb_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (  q1, hadw1_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet (  q2, hadw2_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet ( hb1, higgs_label, jet_res));
  ev.add_jet (Lepjets_Event_Jet ( hb2, higgs_label, jet_res));

  ev.met() = nu;
  ev.kt_res() = kt_res;

  // Simulate SVX tagging.
  tagsim (args, ev, engine);

  // Smear the event, if requested.
  if (args.smear())
    ev.smear (engine, args.smear_dir());

  // Done!
  return ev;
}


} // namespace hitfit
