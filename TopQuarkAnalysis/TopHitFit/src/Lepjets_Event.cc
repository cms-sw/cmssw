//
// $Id: Lepjets_Event.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Lepjets_Event.h
// Purpose: Represent a simple `event' consisting of leptons and jets.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Lepjets_Event.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event.cc

    @brief Represent a simple event consisting of lepton(s) and jet(s).
    See the documentation of header file Lepjets_Event.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include <algorithm>
#include <functional>
#include <cmath>
#include <cassert>


using std::stable_sort;
using std::less;
using std::not2;
using std::remove_if;
using std::abs;


namespace hitfit {


Lepjets_Event::Lepjets_Event (int runnum, int evnum)
//
// Purpose: Constructor.
//
// Inputs:
//   runnum -      The run number.
//   evnum -       The event number.
//
  : _zvertex (0),
    _isMC(false),
    _runnum (runnum),
    _evnum (evnum),
    _dlb (-1),
    _dnn (-1)
{
}


const int& Lepjets_Event::runnum () const
//
// Purpose: Return the run number.
//
// Returns:
//   The run number.
//
{
  return _runnum;
}


int& Lepjets_Event::runnum ()
//
// Purpose: Return the run number.
//
// Returns:
//   The run number.
//
{
  return _runnum;
}


const int& Lepjets_Event::evnum () const
//
// Purpose: Return the event number.
//
// Returns:
//   The event number.
//
{
  return _evnum;
}


int& Lepjets_Event::evnum ()
//
// Purpose: Return the event number.
//
// Returns:
//   The event number.
//
{
  return _evnum;
}


std::vector<Lepjets_Event_Lep>::size_type Lepjets_Event::nleps () const
//
// Purpose: Return the length of the lepton list.
//
// Returns:
//   The length of the lepton list.
//
{
  return _leps.size ();
}


std::vector<Lepjets_Event_Jet>::size_type Lepjets_Event::njets () const
//
// Purpose: Return the length of the jet list.
//
// Returns:
//   The length of the jet list.
//
{
  return _jets.size ();
}


Lepjets_Event_Lep& Lepjets_Event::lep (std::vector<Lepjets_Event_Lep>::size_type  i)
//
// Purpose: Return the Ith lepton.
//
// Inputs:
//   i -           The lepton index (counting from 0).
//
// Returns:
//   The Ith lepton.
//
{
  assert (i < _leps.size());
  return _leps[i];
}


Lepjets_Event_Jet& Lepjets_Event::jet (std::vector<Lepjets_Event_Jet>::size_type i)
//
// Purpose: Return the Ith jet.
//
// Inputs:
//   i -           The jet index (counting from 0).
//
// Returns:
//   The Ith jet.
//
{
  assert (i < _jets.size());
  return _jets[i];
}


const Lepjets_Event_Lep& Lepjets_Event::lep (std::vector<Lepjets_Event_Lep>::size_type i) const
//
// Purpose: Return the Ith lepton.
//
// Inputs:
//   i -           The lepton index (counting from 0).
//
// Returns:
//   The Ith lepton.
//
{
  assert (i < _leps.size());
  return _leps[i];
}


const Lepjets_Event_Jet& Lepjets_Event::jet (std::vector<Lepjets_Event_Jet>::size_type i) const
//
// Purpose: Return the Ith jet.
//
// Inputs:
//   i -           The jet index (counting from 0).
//
// Returns:
//   The Ith jet.
//
{
  assert (i < _jets.size());
  return _jets[i];
}


Fourvec& Lepjets_Event::met ()
//
// Purpose: Return the missing Et.
//
// Returns:
//   The missing Et.
//
{
  return _met;
}


const Fourvec& Lepjets_Event::met () const
//
// Purpose: Return the missing Et.
//
// Returns:
//   The missing Et.
//
{
  return _met;
}


Resolution& Lepjets_Event::kt_res ()
//
// Purpose: Return the kt resolution.
//
// Returns:
//   The kt resolution.
//
{
  return _kt_res;
}


const Resolution& Lepjets_Event::kt_res () const
//
// Purpose: Return the kt resolution.
//
// Returns:
//   The kt resolution.
//
{
  return _kt_res;
}


double Lepjets_Event::zvertex () const
//
// Purpose: Return the z-vertex.
//
// Returns:
//   The z-vertex.
//
{
  return _zvertex;
}


double& Lepjets_Event::zvertex ()
//
// Purpose: Return the z-vertex.
//
// Returns:
//   The z-vertex.
//
{
  return _zvertex;
}


bool Lepjets_Event::isMC () const
//
// Purpose: Return the isMC flag.
//
// Returns:
//   The isMC flag.
//
{
  return _isMC;
}


void Lepjets_Event::setMC (bool isMC)
//
// Purpose: set isMC flag.
//
// Returns:
//   nothing
//
{
  _isMC = isMC;
}

double Lepjets_Event::dlb () const
//
// Purpose: Return the LB discriminant.
//
// Returns:
//   The LB discriminant.
//
{
  return _dlb;
}


double& Lepjets_Event::dlb ()
//
// Purpose: Return the LB discriminant.
//
// Returns:
//   The LB discriminant.
//
{
  return _dlb;
}


double Lepjets_Event::dnn () const
//
// Purpose: Return the NN discriminant.
//
// Returns:
//   The NN discriminant.
//
{
  return _dnn;
}


double& Lepjets_Event::dnn ()
//
// Purpose: Return the NN discriminant.
//
// Returns:
//   The NN discriminant.
//
{
  return _dnn;
}


Fourvec Lepjets_Event::sum (int type) const
//
// Purpose: Sum all objects with type code TYPE.
//
// Inputs:
//   type -        The type code to match.
//
// Returns:
//   The sum of all objects with type code TYPE.
//
{
  Fourvec out;
  for (std::vector<Lepjets_Event_Lep>::size_type  i=0; i < _leps.size(); i++)
    if (_leps[i].type() == type)
      out += _leps[i].p();
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < _jets.size(); i++)
    if (_jets[i].type() == type)
      out += _jets[i].p();
  return out;
}


Fourvec Lepjets_Event::kt () const
//
// Purpose: Calculate kt --- sum of all objects plus missing Et.
//
// Returns:
//   The event kt.
{
  Fourvec v = _met;
  for (std::vector<Lepjets_Event_Lep>::size_type i=0; i < _leps.size(); i++)
    v += _leps[i].p();
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < _jets.size(); i++)
    v += _jets[i].p();
  return v;
}


void Lepjets_Event::add_lep (const Lepjets_Event_Lep& lep)
//
// Purpose: Add a lepton to the event.
//
// Inputs:
//   lep -         The lepton to add.
//
{
  _leps.push_back (lep);
}


void Lepjets_Event::add_jet (const Lepjets_Event_Jet& jet)
//
// Purpose: Add a jet to the event.
//
// Inputs:
//   jet -         The jet to add.
//
{
  _jets.push_back (jet);
}


void Lepjets_Event::smear (CLHEP::HepRandomEngine& engine, bool smear_dir /*= false*/)
//
// Purpose: Smear the objects in the event according to their resolutions.
//
// Inputs:
//   engine -      The underlying RNG.
//   smear_dir -   If false, smear the momentum only.
//
{
  Fourvec before, after;
  for (std::vector<Lepjets_Event_Lep>::size_type i=0; i < _leps.size(); i++) {
    before += _leps[i].p();
    _leps[i].smear (engine, smear_dir);
    after += _leps[i].p();
  }
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < _jets.size(); i++) {
    before += _jets[i].p();
    _jets[i].smear (engine, smear_dir);
    after += _jets[i].p();
  }

  Fourvec kt = _met + before;
  kt(Fourvec::X) = _kt_res.pick (kt(Fourvec::X), kt(Fourvec::X), engine);
  kt(Fourvec::Y) = _kt_res.pick (kt(Fourvec::Y), kt(Fourvec::Y), engine);
  _met = kt - after;
}


void Lepjets_Event::sort ()
//
// Purpose: Sort the objects in the event in order of descending pt.
//
{
  std::stable_sort (_leps.begin(), _leps.end(), not2 (less<Lepjets_Event_Lep> ()));
  std::stable_sort (_jets.begin(), _jets.end(), not2 (less<Lepjets_Event_Lep> ()));
}


std::vector<int> Lepjets_Event::jet_types() const
//
// Purpose: Return the jet types of the event
//
{
  std::vector<int> ret;
  for (std::vector<Lepjets_Event_Jet>::size_type ijet =  0 ;
       ijet != _jets.size() ;
       ijet++) {
    ret.push_back(jet(ijet).type());
  }
  return ret;
}


bool Lepjets_Event::set_jet_types(const std::vector<int>& _jet_types)
//
// Purpose: Set the jet types of the event
// Return false if it fails, trus if it succeeds
//
{
  if (_jets.size() != _jet_types.size()) {
    return false;
  }
  bool saw_hadw1 = false;
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < njets(); i++) {
    int t = _jet_types[i];
    if (t == hadw1_label) {
      if (saw_hadw1)
        t = hadw2_label;
      saw_hadw1 = true;
    }
    jet (i).type() = t;
  }
  return true;
}


namespace {


struct Lepjets_Event_Cutter
//
// Purpose: Helper for cutting on objects.
//
{
  Lepjets_Event_Cutter (double pt_cut, double eta_cut)
    : _pt_cut (pt_cut), _eta_cut (eta_cut)
  {}
  bool operator() (const Lepjets_Event_Lep& l) const;
  double _pt_cut;
  double _eta_cut;
};


bool Lepjets_Event_Cutter::operator () (const Lepjets_Event_Lep& l) const
//
// Purpose: Object cut evaluator.
//
{
  return ! (l.p().perp() > _pt_cut && abs (l.p().pseudoRapidity()) < _eta_cut);
}


} // unnamed namespace


int Lepjets_Event::cut_leps (double pt_cut, double eta_cut)
//
// Purpose: Remove all leptons failing the pt and eta cuts.
//
// Inputs:
//   pt_cut -      Pt cut.  Remove objects with pt less than this.
//   eta_cut -     Eta cut.  Remove objects with abs(eta) larger than this.
//
// Returns:
//   The number of leptons remaining after the cuts.
//
{
  _leps.erase (remove_if (_leps.begin(), _leps.end(),
                          Lepjets_Event_Cutter (pt_cut, eta_cut)),
               _leps.end ());
  return _leps.size ();
}


int Lepjets_Event::cut_jets (double pt_cut, double eta_cut)
//
// Purpose: Remove all jets failing the pt and eta cuts.
//
// Inputs:
//   pt_cut -      Pt cut.  Remove objects with pt less than this.
//   eta_cut -     Eta cut.  Remove objects with abs(eta) larger than this.
//
// Returns:
//   The number of jets remaining after the cuts.
//
{
  _jets.erase (remove_if (_jets.begin(), _jets.end(),
                          Lepjets_Event_Cutter (pt_cut, eta_cut)),
               _jets.end ());
  return _jets.size ();
}


void Lepjets_Event::trimjets (std::vector<Lepjets_Event_Jet>::size_type n)
//
// Purpose: Remove all but the first N jets.
//
// Inputs:
//   n -           The number of jets to keep.
//
{
  if (n >= _jets.size())
    return;
  _jets.erase (_jets.begin() + n, _jets.end());
}


std::ostream& Lepjets_Event::dump (std::ostream& s, bool full /*=false*/) const
//
// Purpose: Dump out this object.
//
// Inputs:
//   s -           The stream to which to write.
//   full -        If true, dump all information for this object.
//
// Returns:
//   The stream S.
//
{
  s << "Run: " << _runnum << "  Event: " << _evnum << "\n";
  s << "Leptons:\n";
  for (std::vector<Lepjets_Event_Lep>::size_type i=0; i < _leps.size(); i++) {
    s << "  ";
    _leps[i].dump (s, full);
    s << "\n";
  }
  s << "Jets:\n";
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < _jets.size(); i++) {
    s << "  ";
    _jets[i].dump (s, full);
    s << "\n";
  }
  s << "Missing Et: " << _met << "\n";
  if (_zvertex != 0)
    s << "z-vertex: " << _zvertex << "\n";
  return s;
}


std::string Lepjets_Event::jet_permutation() const
//
// Purpose: Return a string representation of the jet permutation
//          g - isr/gluon
//          b - leptonic b
//          B - hadronic b
//          w - hadronic W
//          h - Higgs to b-bbar
//          ? - Unknown
{
    std::string permutation;
    for (size_t jet = 0 ; jet != _jets.size() ; ++jet) {
        permutation = permutation + hitfit::jetTypeString(_jets[jet].type());
    }
    return permutation;
}

/**
    @brief Output stream operator, print the content of this Lepjets_Event
    to an output stream.

    @param s The output stream to which to write.

    @param ev The instance of Lepjets_Event to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Lepjets_Event& ev)
//
// Purpose: Dump out this object.
//
// Inputs:
//   s -           The stream to which to write.
//   ev -          The object to dump.
//
// Returns:
//   The stream S.
//
{
  return ev.dump (s);
}


} // namespace hitfit





