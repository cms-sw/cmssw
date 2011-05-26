//
// $Id: run1_ascii_io.cc,v 1.4 2010/04/13 19:25:16 haryo Exp $
//
// File: src/run1_ascii_io.cc
// Purpose: Read and write the run 1 ntuple dump files.
// Created: Dec, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/run1_ascii_io.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file run1_ascii_io.cc

    @brief Source file for legacy code to read and write D0 Run I
    ntuple file. No detailed doxygen documentation will
    be provided.

 */

#include "TopQuarkAnalysis/TopHitFit/interface/run1_ascii_io.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <istream>
#include <ostream>
#include <cctype>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <cstdlib>

using std::cerr;
using std::cout;
using std::istream;
using std::ostream;
using std::isspace;
using std::string;
using std::vector;
using std::abs;
using std::getline;
using std::istringstream;


namespace hitfit {


//*************************************************************************
// Argument handling.
//


Run1_Ascii_IO_Args::Run1_Ascii_IO_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _jet_type_wanted (defs.get_string ("jet_type_wanted")[0]),
    _ele_res (defs.get_string ("ele_res_str")),
    _muo_res (defs.get_string ("muo_res_str")),
    _jet_res (defs.get_string ("jet_res_str")),
    _kt_res (defs.get_string ("kt_res_str"))
{
}


char Run1_Ascii_IO_Args::jet_type_wanted () const
//
// Purpose: Return the jet_type_wanted parameter.
//          See the header for documentation.
//
{
  return _jet_type_wanted;
}


const Vector_Resolution& Run1_Ascii_IO_Args::ele_res () const
//
// Purpose: Return the ele_res parameter.
//          See the header for documentation.
//
{
  return _ele_res;
}


const Vector_Resolution& Run1_Ascii_IO_Args::muo_res () const
//
// Purpose: Return the muo_res parameter.
//          See the header for documentation.
//
{
  return _muo_res;
}


const Vector_Resolution& Run1_Ascii_IO_Args::jet_res () const
//
// Purpose: Return the jet_res parameter.
//          See the header for documentation.
//
{
  return _jet_res;
}


const Resolution& Run1_Ascii_IO_Args::kt_res () const
//
// Purpose: Return the kt_res parameter.
//          See the header for documentation.
//
{
  return _kt_res;
}


//*************************************************************************
// Event reading.
//


namespace {


Fourvec get_fourvec (istream& s)
//
// Purpose: Read a 4-vector from stream S.
//
// Inputs:
//   s -           The stream from which to read.
//
// Returns:
//  The 4-vector.
//
{
  double r[4];
  s >> r[0] >> r[1] >> r[2] >> r[3];
  return Fourvec (r[0], r[1], r[2], r[3]);
}


bool get_svx_tag (istream& s)
//
// Purpose: See if there is a SVX tag indicator in the input stream.
//
// Inputs:
//   s -           The stream from which to read.
//
// Returns:
//   True if we saw a SVX tag, false otherwise.
//
{
  char c = ' ';
  while (s.get (c) && c == ' ')
    ;
  s.putback (c);
  if (c == '\n')
    return false;

  char buf[20];
  s.width (sizeof (buf)-1);
  s >> buf;
  s.width (0);

  if (string (buf) == "svx")
    return true;

  return false;
}


void skipline (istream& s)
//
// Purpose: Skip the rest of the current line.
//
// Inputs:
//   s -           The stream from which to read.
//
{
  char c;
  while (s.get (c) && c != '\n')
    ;
}


string read_label (istream& s)
//
// Purpose: Read the next 2-character label from the input stream.
//
// Inputs:
//   s -           The stream from which to read.
//
// Returns:
//   The next label, or an empty string at the end of the event.
//
{
  char lab[3];
  char c = ' ';

  // Skip past a line end.
  while (s.get (c) && c == ' ')
    ;

  if (c != '\n') {
    s.putback (c);
  }

  s.get (c);
  if (c == '\n' || isspace (c))
    return "";

  // Get the label.
  lab[0] = c;
  s.get (c);

  if (c == '\n') {
    cerr << "bad file format\n";
    abort ();
  }

  lab[1] = c;

  lab[2] = '\0';

  return lab;
}


vector<int> read_jetperm (istream& s, int njets)
//
// Purpose: Read a jet permutation from the input stream.
//
// Inputs:
//   s -           The stream from which to read.
//   njets -       The number of jets for this event.
//
// Returns:
//   The jet permutation.
//
{
  vector<int> types (njets);

  for (int i=0; i < njets; i++) {
    char c;
    s >> c;
    if ( ! s.good () || c == '\n') {
      cerr << "inconsistent jet count\n";
      abort ();
    }


    switch (c) {
    case 'b':  types[i] = lepb_label;    break;
    case 'B':  types[i] = hadb_label;    break;
    case 'w':  types[i] = hadw1_label;   break;
    case 'W':  types[i] = hadw2_label;   break;
    case 'i':  types[i] = isr_label;     break;
    case 'h':  types[i] = higgs_label;   break;
    default :  types[i] = unknown_label; break;
    }
  }

  return types;
}


} // unnamed namespace


Lepjets_Event read_run1_ascii (istream& s,
                               const Run1_Ascii_IO_Args& args)
//
// Purpose: Read an event from stream S.
//
// Inputs:
//   s -           The stream from which to read.
//   args -        Parameter settings.
//
// Retuns:
//   The event read.
//   Returns an event with runnum==-1 and evnum==-1 at EOF.
//
{
  if (read_label (s) != "ev") {
    if (!s || s.eof())
      return Lepjets_Event (-1, -1);
    cerr << "bad file format";
    abort ();
  }

  // Read the header information.
  int runnum;
  int evnum;
  int dumi;
  double dumf;
  s >> evnum;
  s >> runnum;
  s >> dumi;  // XXX evinum1
  s >> dumi;  // XXX evinum2
  skipline (s);

  Lepjets_Event ev (runnum, evnum);

  vector<int> jet_types;
  unsigned types_ptr = 0;
  Fourvec tag;
  double tag_edep = 0;
  int bjet_ndx = 0;

  // Loop over tags.
  string line;
  while (getline (s, line)) {
    if (line.size() == 0)
      break;
    istringstream sline (line);
    string lab = read_label (sline);
    if (lab.size() == 0)
      break;

    if (lab == "//") {
      cout << line.substr(2) << "\n";
    }

    else if (lab == "el") {
      ev.add_lep (Lepjets_Event_Lep (get_fourvec (sline),
                                     electron_label,
                                     args.ele_res ()));
    }

    else if (lab == "mu") {
      ev.add_lep (Lepjets_Event_Lep (get_fourvec (sline),
                                     muon_label,
                                     args.muo_res ()));
    }

    else if (lab == "nu")
      ev.met() = get_fourvec (sline);

    else if (lab == "ta")
      tag = get_fourvec (sline);

    else if (lab == "mc") {
      sline >> dumi;
      ev.setMC (dumi != 0);
    }

    else if (lab == "k1") {
      sline >> dumf; // XXX saved_ht;
      sline >> dumf; // XXX saved_aplan;
      sline >> ev.zvertex();
      sline >> dumf; // XXX saved_aplanw;
    }

    else if (lab == "k2") {
      sline >> dumf; // XXX metcut
      sline >> dumf; // XXX aplcut
      sline >> dumf; // XXX etatcut[0]
      sline >> dumf; // XXX etatcut[1]
      sline >> dumf; // XXX htcut
      sline >> dumi; // XXX okcut
    }

    else if (lab == "k3") {
      sline >> dumf; // XXX probtri
      sline >> dumf; // XXX trinnpb
      sline >> dumi; // XXX oktri
      sline >> dumf; // XXX trinnpu
    }

    else if (lab == "k4") {
      sline >> dumf; // XXX pts_lep
      sline >> dumf; // XXX pts_nu
    }

    else if (lab == "k5") {
      char buf[80];
      sline.get (buf, sizeof (buf)-1);
      for (unsigned int i=0; i<sizeof (buf) && buf[i] != '\0'; i++) {
        switch (buf[i]) {
        case 'h':
          // XXX ev._sav.hot_cell_flag = true;
          break;

        case 'g':
          // XXX ev._sav.gamma_flag = true;
          break;

        case 'm':
          // XXX ev._sav.mrbs_flag = true;
          break;
        }
      }
    }

    else if (lab == "k6") {
      sline >> dumi; // XXX nj15_2_5
      sline >> dumi; // XXX nj15_2
      sline >> dumi; // XXX nj20_2
    }

    else if (lab == "k7") {
      double sk2_mu;

      sline >> dumf; // XXX eta_w
      sline >> sk2_mu;
      sline >> dumf; // XXX smet_xx
      sline >> dumf; // XXX smet_yy
      sline >> dumf; // XXX smet_xy

      for (std::vector<Lepjets_Event_Lep>::size_type j=0; j<ev.nleps(); j++) {
        if (ev.lep(j).type() == muon_label) {
          ev.lep(j).res() = Vector_Resolution (Resolution (sqrt (sk2_mu),true),
                                               args.muo_res().eta_res (),
                                               args.muo_res().phi_res ());
        }
      }
    }

    else if (lab == "k8") {
      sline >> tag_edep;
    }

    else if (lab == "k9") {
      sline >> dumf; // XXX mtl4j
      sline >> dumf; // XXX ht2
    }

    else if (lab == "ka") {
      for (int i=0; i < 7; i++)
        sline >> dumf; // XXX triargs
    }

    else if (lab == "id") {
      getline (s, line);
      getline (s, line);
      istringstream sline1 (line);
      sline1 >> dumf;
      sline1 >> dumf;
      sline1 >> dumf;
      sline1 >> dumf;
      sline1 >> ev.dlb();
      sline1 >> ev.dnn();
      getline (s, line);
    }

    else if (lab[0] == 'k')
      ;

    else if (lab[0] == 'h' && lab[1] == args.jet_type_wanted()) {
      int njets;
      sline >> njets;
      sline >> bjet_ndx;
      jet_types = read_jetperm (sline, njets);
      types_ptr = 0;
    }

    else if (lab[0] == 'h')
      ;

    else if (lab[0] == 'j' && lab[1] == args.jet_type_wanted()) {
      // XXX should really use etadep resolutions
      assert (ev.njets() < jet_types.size());
      Fourvec j = get_fourvec (sline);
      bool svx_flag = get_svx_tag (sline);
      ev.add_jet (Lepjets_Event_Jet (j,
                                     jet_types[ev.njets()],
                                     args.jet_res (),
                                     svx_flag));
    }

    else if (lab[0] == 'j')
      ;

    else
      cerr << "Warning: unknown label " << lab << "\n";

    skipline (sline);
  }

  // XXX sign of index is tag sign.
  bjet_ndx = abs (bjet_ndx);
  if (bjet_ndx < 0) {
      bjet_ndx = -1*bjet_ndx;
  }
  if (bjet_ndx > 0) {
    assert (bjet_ndx <= int(ev.njets()));
    ev.jet (bjet_ndx-1).tag_lep() = tag;
    ev.jet (bjet_ndx-1).slt_tag() = true;
    ev.jet (bjet_ndx-1).slt_edep() = tag_edep;
  }

  ev.kt_res() = args.kt_res();

  return ev;
}


//*************************************************************************
// Event writing
//


namespace {


ostream& put_fourvec (ostream& s, const Fourvec& v)
//
// Purpose: Print 4-vector V to S.
//
// Inputs:
//   s -           The stream to which to write.
//   v -           The vector to write.
//
// Returns:
//   The stream S.
//
{
  s << v.x() << " " << v.y() << " " << v.z() << " " << v.e();
  return s;
}


ostream& put_jetperm (ostream& s, const Lepjets_Event& ev)
//
// Purpose: Print jet permutation info for EV to S.
//
// Inputs:
//   s -           The stream to which to write.
//   ev -          The event for which we should write information.
//
// Returns:
//   The stream S.
//
{
  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < ev.njets(); i++) {
    char c;
    switch (ev.jet(i).type()) {
    case lepb_label:  c = 'b'; break;
    case hadb_label:  c = 'B'; break;
    case hadw1_label: c = 'w'; break;
    case hadw2_label: c = 'W'; break;
    case isr_label:   c = 'i'; break;
    case higgs_label: c = 'h'; break;
    default:          c = '.'; break;
    }
    s << c;
  }
  return s;
}


} // unnamed namespace


ostream& write_run1_ascii (std::ostream& s, const Lepjets_Event& ev)
//
// Purpose: Print event EV to S.
//
// Inputs:
//   s -           The stream to which to write.
//   ev -          The event to write.
//
// Returns:
//   The stream S.
//
{
  // XXX no mu tag info

  s << "ev " << ev.evnum() << " " << ev.runnum() << " 1 1\n";
  for (std::vector<Lepjets_Event_Lep>::size_type i=0; i < ev.nleps(); i++) {
    if (ev.lep(i).type() == muon_label)
      s << "mu ";
    else
      s << "el ";
    put_fourvec (s, ev.lep(i).p()) << "\n";
  }

  s << "nu ";
  put_fourvec (s, ev.met()) << "\n";

  s << "h5  " << ev.njets() << " 0 ";
  put_jetperm (s, ev) << "\n";

  for (std::vector<Lepjets_Event_Jet>::size_type i=0; i < ev.njets(); i++) {
    s << "j5 ";
    put_fourvec (s, ev.jet(i).p());
    if (ev.jet(i).svx_tag())
      s << "  svx";
    s << "\n";
  }

  s << "\n";

  return s;
}


} // namespace hitfit
