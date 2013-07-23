//
// $Id: Vector_Resolution.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Vector_Resolution.cc
// Purpose: Calculate resolutions in p, phi, eta.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Vector_Resolution.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Vector_Resolution.cc

    @brief Calculate and represent resolution for a vector
    of momentum \f$p\f$, pseudorapidity \f$\eta\f$, and azimuthal
    angle \f$\phi\f$. See the documentation for the header file
    Vector_Resolution.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <cmath>
#include <ostream>
#include <cctype>


using std::ostream;
using std::string;
using std::isspace;


namespace {

/**
    @brief Helper function: extract the <i>i</i>-th slash-separated field
    from a string.

    @param s The string to analyze.

    @param i The field number, starting from 0.
 */
string field (string s, int i)
//
// Purpose: Extract the Ith slash-separated field from S.
//
// Inputs:
//   s -           The string to analyze.
//   i -           Field number (starting with 0).
//
// Returns:
//   Field I of S.
//
{
  string::size_type pos = 0;
  while (i > 0) {
    pos = s.find ('/', pos);
    if (pos == string::npos)
      return "";
    ++pos;
    --i;
  }

  string::size_type pos2 = s.find ('/', pos+1);
  if (pos2 != string::npos)
    pos2 = pos2 - pos;

  while (pos < pos2 && isspace (s[pos]))
    ++pos;

  return s.substr (pos, pos2);
}


} // unnamed namespace


namespace hitfit {


Vector_Resolution::Vector_Resolution ()
//
// Purpose: Default constructor.  Create a vector resolution object with
//          infinite precision.v
//
  : _p_res   ("0,0,0"),
    _eta_res ("0,0,0"),
    _phi_res ("0,0,0"),
    _use_et  (false)

{
}


Vector_Resolution::Vector_Resolution (std::string s)
//
// Purpose: Constructor.
//
// Inputs:
//   s -           String encoding the resolution parameters, as described
//                 in the comments in the header.
//
  : _p_res   (field (s, 0)),
    _eta_res (field (s, 1)),
    _phi_res (field (s, 2)),
    _use_et  (field (s, 3) == "et")
{
}


Vector_Resolution::Vector_Resolution (const Resolution& p_res,
                                      const Resolution& eta_res,
                                      const Resolution& phi_res,
                                      bool use_et /*= false*/)
//
// Purpose: Constructor from individual resolution objects.
//
// Inputs:
//   p_res -       Momentum resolution object.
//   eta_res -     Eta resolution object.
//   phi_res -     Phi resolution object.
//   use_et -      If true, momentum resolution is based on pt, not p.
//
  : _p_res   (p_res),
    _eta_res (eta_res),
    _phi_res (phi_res),
    _use_et  (use_et)
{
}


const Resolution& Vector_Resolution::p_res () const
//
// Purpose: Return the momentum resolution object.
//
// Returns:
//   The momentum resolution object.
//
{
  return _p_res;
}


const Resolution& Vector_Resolution::eta_res () const
//
// Purpose: Return the eta resolution object.
//
// Returns:
//   The eta resolution object.
//
{
  return _eta_res;
}


const Resolution& Vector_Resolution::phi_res () const
//
// Purpose: Return the phi resolution object.
//
// Returns:
//   The phi resolution object.
//
{
  return _phi_res;
}


bool Vector_Resolution::use_et () const
//
// Purpose: Return the use_et flag.
//
// Returns:
//   The use_et flag.
//
{
  return _use_et;
}


namespace {


/**
    @brief Helper function: calculate the uncertainty/resolution \f$\sigma\f$
    for the momentum of a four-momentum.

    @param v The four-momentum.

    @param res The resolution object.

    @param use_et If <b>TRUE</b> then use \f$p_{T}\f$ instead of \f$p\f$
    for momentum resolution.
 */
double find_sigma (const Fourvec& v, const Resolution& res, bool use_et)
//
// Purpose: Helper for *_sigma functions below.
//
// Inputs:
//   v -           The 4-vector.
//   res -         The resolution object.
//   use_et -      Use_et flag.
//
// Returns:
//   The result of res.sigma() (not corrected for e/et difference).
//
{
  double ee = use_et ? v.perp() : v.e(); // ??? is perp() correct here?
  return res.sigma (ee);
}


} // unnamed namespace


double Vector_Resolution::p_sigma (const Fourvec& v) const
//
// Purpose: Calculate momentum resolution for 4-vector V.
//
// Inputs:
//   v -           The 4-vector.
//
// Returns:
//   The momentum resolution for 4-vector V.
//
{
  double sig = find_sigma (v, _p_res, _use_et);
  if (_use_et) {
    if(_p_res.inverse()){
      sig *= v.perp() / v.e();
    }else{
      sig *= v.e() / v.perp();
    }
  }
  return sig;
}


double Vector_Resolution::eta_sigma (const Fourvec& v) const
//
// Purpose: Calculate eta resolution for 4-vector V.
//
// Inputs:
//   v -           The 4-vector.
//
// Returns:
//   The eta resolution for 4-vector V.
//
{
  return find_sigma (v, _eta_res, _use_et);
}


double Vector_Resolution::phi_sigma (const Fourvec& v) const
//
// Purpose: Calculate phi resolution for 4-vector V.
//
// Inputs:
//   v -           The 4-vector.
//
// Returns:
//   The phi resolution for 4-vector V.
//
{
  return find_sigma (v, _phi_res, _use_et);
}



namespace {


/**
    @brief Helper function: smear the pseudorapidity of a four-momentum

    @param v The four-momentum to smear.

    @param ee The energy for \f$sigma\f$ calculation.

    @param res The resolution object, giving the amount of smearing.

    @param engine The underlying random number generator.
 */
void smear_eta (Fourvec& v, double ee,
                const Resolution& res, CLHEP::HepRandomEngine& engine)
//
// Purpose: Smear the eta direction of V.
//
// Inputs:
//   v -           The 4-vector to smear.
//   ee -          Energy for sigma calculation.
//   res -         The resolution object, giving the amount of smearing.
//   engine -      The underlying RNG.
//
// Outputs:
//   v -           The smeared 4-vector.
//
{
  double rot = res.pick (0, ee, engine);
  roteta (v, rot);
}


/**
    @brief Helper function: smear the azimuthal angle of a four-momentum

    @param v The four-momentum to smear.

    @param ee The energy for \f$sigma\f$ calculation.

    @param res The resolution object, giving the amount of smearing.

    @param engine The underlying random number generator.
 */
void smear_phi (Fourvec& v, double ee,
                const Resolution& res, CLHEP::HepRandomEngine& engine)
//
// Purpose: Smear the phi direction of V.
//
// Inputs:
//   v -           The 4-vector to smear.
//   ee -          Energy for sigma calculation.
//   res -         The resolution object, giving the amount of smearing.
//   engine -      The underlying RNG.
//
// Outputs:
//   v -           The smeared 4-vector.
// 
{
  double rot = res.pick (0, ee, engine);
  v.rotateZ (rot);
}


} // unnamed namespace


void Vector_Resolution::smear (Fourvec& v,
                               CLHEP::HepRandomEngine& engine,
                               bool do_smear_dir /*= false*/) const
//
// Purpose: Smear a 4-vector according to the resolutions.
//
// Inputs:
//   v -           The 4-vector to smear.
//   engine -      The underlying RNG.
//   do_smear_dir- If false, only smear the energy.
//
// Outputs:
//   v -           The smeared 4-vector.
//
{
  double ee = _use_et ? v.perp() : v.e(); // ??? is perp() correct?
  v *= _p_res.pick (ee, ee, engine) / ee;

  if (do_smear_dir) {
    smear_eta (v, ee, _eta_res, engine);
    smear_phi (v, ee, _phi_res, engine);
  }
}


/**
    @brief Output stream operator, print the content of this Vector_Resolution
    object to an output stream.

    @param s The stream to which to write.

    @param r The instance of Vector_Resolution to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Vector_Resolution& r)
//
// Purpose: Dump this object to S.
//
// Inputs:
//    s -          The stream to which to write.
//    r -          The object to dump.
//
// Returns:
//   The stream S.
//   
{
  s << r._p_res << "/ " << r._eta_res << "/ " << r._phi_res;
  if (r._use_et)
    s << "/et";
  s << "\n";
  return s;
}


} // namespace hitfit
