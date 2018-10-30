#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

int HFFibreFiducial::PMTNumber(const G4ThreeVector& pe_effect)
{
  double xv  = pe_effect.x();          // X in global system
  double yv  = pe_effect.y();          // Y in global system
  double phi = atan2(yv, xv);          // In global system
  if (phi < 0.) phi+=CLHEP::pi;        // Just for security
  double dph = CLHEP::pi/18;           // 10 deg = a half sector width
  double sph = dph+dph;                // 20 deg = a sector width
  int   nphi = phi/dph;                // 10 deg sector #
  LogDebug("HFShower") <<"HFFibreFiducial:***> P = " << pe_effect 
		       << ", phi = " << phi/CLHEP::deg;
  if (nphi > 35) nphi=35;              // Just for security
  double xl=0.;                        // local sector coordinates (left/right)
  double yl=0.;                        // local sector coordinates (down/up)
  int nwid=0;                          // CMS widget number (@@ not used now M.K.)
  double phir= 0.;                     // phi for rotation to the sector system
  if (nphi==0 || nphi==35)
  {
    yl=xv;
    xl=yv;
    nwid=6;
  }
  else if (nphi==17 || nphi==18)
  {
    yl=-xv;
    xl=-yv;
    nwid=15;
    phir=CLHEP::pi;                    // nr=9 ?
  }
  else
  {
    int nr = (nphi+1)/2;               // a sector # (@@ internal definition)
    nwid = 6-nr;
    if(nwid <= 0) nwid+=18;            // @@ +z || -z M.K. to be improved
    phir= sph*nr;                      // nontrivial phi for rotation to the sector system
    double cosr= cos(phir);
    double sinr= sin(phir);
    yl= xv*cosr+yv*sinr;
    xl= yv*cosr-xv*sinr;
    LogDebug("HFShower") << "HFFibreFiducial: nr " << nr << " phi " 
			 << phir/CLHEP::deg;
  }
  if (yl < 0) yl =-yl;
  LogDebug("HFShower") << "HFFibreFiducial: Global Point " << pe_effect
		       << " nphi " << nphi 
		       << " Local Sector Coordinates (" 
		       << xl << ", " << yl << "), widget # " << nwid;
  // Provides a PMT # for the (x,y) hit in the widget # nwid (M. Kosov, 11.2010)
  // Send comments/questions to Mikhail.Kossov@cern.ch
  // nwid = 1-18 for Forward HF, 19-36 for Backward HF (all equal now)
  // npmt = 0 for No Hit, 1-24 for H(Long) PMT, 25-48 for E(Short) PMT, negative for souces

  static const int nWidM=36;
  if (nwid > nWidM || nwid <= 0)   {
    LogDebug("HFShower") << "-Warning-HFFibreFiducial::PMTNumber: "
				 << nwid << " == wrong widget number";
    return 0;
  }
  static const double yMin= 13.1*CLHEP::cm; // start of the active area (Conv to mm?)
  static const double yMax=129.6*CLHEP::cm; // finish of the active area (Conv to mm?)
  if( yl < yMin || yl >= yMax ) {
    LogDebug("HFShower") << "-Warning-HFFibreFiducial::PMTNumber: "
				 << "Point with y = " << yl 
				 << " outside acceptance [" << yMin << ":" 
				 << yMax << "],  X = " << xv << ", Y = " 
				 << yv << ", x = " << xl << ", nW = " 
				 << nwid << ", phi = " << phi/CLHEP::deg 
				 << ", phir = " << phir/CLHEP::deg;
    return 0;                     // ===> out of the acceptance
  }
  bool left=true;                 // flag of the left part of the widget
  double r=xl/yl;                 // for the widget acceptance check
  if (r < 0)
  {
    r=-r;
    left=false;
  }
  static const double tg10=.17632698070847; // phi-angular acceptance of the widget
  if (r > tg10) {
    LogDebug("HFShower") <<"-Warning-HFFibreFiducial::PMTNumber: (x = "
				 << xl << ", y = " << yl << ", tg = " << r 
				 << ") out of the widget acceptance tg(10) "
				 << tg10;
    return 0;
  }

  static const int nLay=233;      // a # of the sensetive layers in the widget
  static const int nL001=4;
  static const int nL002=4;
  static const int nL003=5;
  static const int nL004=5;
  static const int nL005=5;  // (5)
  static const int nL006=5;
  static const int nL007=5;
  static const int nL008=6;
  static const int nL009=6;
  static const int nL010=6;  // (6)
  static const int nL011=6;
  static const int nL012=6;
  static const int nL013=6;
  static const int nL014=7;
  static const int nL015=7;
  static const int nL016=7;  // (6)
  static const int nL017=7;
  static const int nL018=7;
  static const int nL019=7;
  static const int nL020=8;
  static const int nL021=8;
  static const int nL022=8;  // (5)
  static const int nL023=8;
  static const int nL024=8;
  static const int nL025=9;
  static const int nL026=9;
  static const int nL027=9;  // (6)
  static const int nL028=9;
  static const int nL029=9;
  static const int nL030=9;
  static const int nL031=10;
  static const int nL032=10;
  static const int nL033=10; // (6)
  static const int nL034=10;
  static const int nL035=10;
  static const int nL036=10;
  static const int nL037=11;
  static const int nL038=11; // (5)
  static const int nL039=11;
  static const int nL040=11;
  static const int nL041=11;
  static const int nL042=12;
  static const int nL043=12;
  static const int nL044=12;
  static const int nL045=12; // (6)
  static const int nL046=12;
  static const int nL047=12;
  static const int nL048=13;
  static const int nL049=13;
  static const int nL050=13; // (6)
  static const int nL051=13;
  static const int nL052=13;
  static const int nL053=13;
  static const int nL054=14;
  static const int nL055=14;
  static const int nL056=14; // (5)
  static const int nL057=14;
  static const int nL058=14;
  static const int nL059=15;
  static const int nL060=15;
  static const int nL061=15; // (6)
  static const int nL062=15;
  static const int nL063=15;
  static const int nL064=15;
  static const int nL065=16;
  static const int nL066=16;
  static const int nL067=16; // (6)
  static const int nL068=16;
  static const int nL069=16;
  static const int nL070=16;
  static const int nL071=17;
  static const int nL072=17;
  static const int nL073=17; // (5)
  static const int nL074=17;
  static const int nL075=17;
  static const int nL076=18;
  static const int nL077=18;
  static const int nL078=18; // (6)
  static const int nL079=18;
  static const int nL080=18;
  static const int nL081=18;
  static const int nL082=19;
  static const int nL083=19; // (6)
  static const int nL084=19;
  static const int nL085=19;
  static const int nL086=19;
  static const int nL087=19;
  static const int nL088=20;
  static const int nL089=20;
  static const int nL090=20; // (5)
  static const int nL091=20;
  static const int nL092=20;
  static const int nL093=21;
  static const int nL094=21;
  static const int nL095=21; // (6)
  static const int nL096=21;
  static const int nL097=21;
  static const int nL098=21;
  static const int nL099=22;
  static const int nL100=22;
  static const int nL101=22; // (6)
  static const int nL102=22;
  static const int nL103=22;
  static const int nL104=22;
  static const int nL105=23;
  static const int nL106=23;
  static const int nL107=23; // (5)
  static const int nL108=23;
  static const int nL109=23;
  static const int nL110=24;
  static const int nL111=24;
  static const int nL112=24; // (6)
  static const int nL113=24;
  static const int nL114=24;
  static const int nL115=24;
  static const int nL116=25;
  static const int nL117=25;
  static const int nL118=25; // (6)
  static const int nL119=25;
  static const int nL120=25;
  static const int nL121=25;
  static const int nL122=26;
  static const int nL123=26;
  static const int nL124=26; // (5)
  static const int nL125=26;
  static const int nL126=26;
  static const int nL127=27;
  static const int nL128=27;
  static const int nL129=27; // (6)
  static const int nL130=27;
  static const int nL131=27;
  static const int nL132=27;
  static const int nL133=28;
  static const int nL134=28;
  static const int nL135=28; // (6)
  static const int nL136=28;
  static const int nL137=28;
  static const int nL138=28;
  static const int nL139=29;
  static const int nL140=29;
  static const int nL141=29; // (5)
  static const int nL142=29;
  static const int nL143=29;
  static const int nL144=30;
  static const int nL145=30;
  static const int nL146=30; // (6)
  static const int nL147=30;
  static const int nL148=30;
  static const int nL149=30;
  static const int nL150=31;
  static const int nL151=31;
  static const int nL152=31; // (6)
  static const int nL153=31;
  static const int nL154=31;
  static const int nL155=31;
  static const int nL156=32;
  static const int nL157=32; // (5)
  static const int nL158=32;
  static const int nL159=32;
  static const int nL160=32;
  static const int nL161=33;
  static const int nL162=33; // (6)
  static const int nL163=33;
  static const int nL164=33;
  static const int nL165=33;
  static const int nL166=33;
  static const int nL167=34;
  static const int nL168=34;
  static const int nL169=34; // (6)
  static const int nL170=34;
  static const int nL171=34;
  static const int nL172=34;
  static const int nL173=35;
  static const int nL174=35;
  static const int nL175=35; // (5)
  static const int nL176=35;
  static const int nL177=35;
  static const int nL178=36;
  static const int nL179=36;
  static const int nL180=36; // (6)
  static const int nL181=36;
  static const int nL182=36;
  static const int nL183=36;
  static const int nL184=37;
  static const int nL185=37;
  static const int nL186=37; // (6)
  static const int nL187=37;
  static const int nL188=37;
  static const int nL189=37;
  static const int nL190=38;
  static const int nL191=38;
  static const int nL192=38; // (5)
  static const int nL193=38;
  static const int nL194=38;
  static const int nL195=39;
  static const int nL196=39;
  static const int nL197=39;
  static const int nL198=39; // (6)
  static const int nL199=39;
  static const int nL200=39;
  static const int nL201=40;
  static const int nL202=40;
  static const int nL203=40; // (6)
  static const int nL204=40;
  static const int nL205=40;
  static const int nL206=40;
  static const int nL207=41;
  static const int nL208=41;
  static const int nL209=41; // (5)
  static const int nL210=41;
  static const int nL211=41;
  static const int nL212=42;
  static const int nL213=42;
  static const int nL214=42;
  static const int nL215=42; // (6)
  static const int nL216=42;
  static const int nL217=42;
  static const int nL218=43;
  static const int nL219=43;
  static const int nL220=43; // (6)
  static const int nL221=43;
  static const int nL222=43;
  static const int nL223=43;
  static const int nL224=44;
  static const int nL225=44;
  static const int nL226=44; // (5)
  static const int nL227=44;
  static const int nL228=44;
  static const int nL229=45;
  static const int nL230=45;
  static const int nL231=45; // (5+1=6)
  static const int nL232=45;
  static const int nL233=45;
  //------------------------------------------------------------------------------------
  // Mean numbers of fibers in the layer is used. In some widgets it's bigger  ***
  // (if the fiber passed throug the hole closer to the edge) and sometimes it ***
  // is smaller (if in some holes of the layer fibers did not pass throug).    ***
  // The real presence of fibers in the holes is now unknown (not documented), ***
  // but the narrow electron showers can be used for revealing of the missing  ***
  // or additional fibers in the widget, because the missing fibers reduce the ***
  // response and additional fibers increas it. So the tables can be improved  ***
  // to be individual for widgets and the FXX/BXX sources-tables can be used.  ***
  // ********************** M.Kosov, Mikhail.Kosssov@cern.ch *********************
  // NNI, NN=tower#(1-24), i=0: dead; i=1: E(L); i=2: H(S); i=3: ESource; i=4: HSource
  static const int tR001[nL001]={132,131,132,131}; // Left Part of the widget (-phi)
  static const int tR002[nL002]={131,132,131,132};
  static const int tR003[nL003]={132,131,132,131,132};
  static const int tR004[nL004]={133,132,131,132,131}; // (5)
  static const int tR005[nL005]={132,131,132,131,132};
  static const int tR006[nL006]={131,132,131,132,131};
  static const int tR007[nL007]={132,131,132,131,132};
  static const int tR008[nL008]={131,132,131,132,131,132}; // _______________________13_
  static const int tR009[nL009]={122,121,122,121,122,121};
  static const int tR010[nL010]={121,122,121,122,123,122}; // (6) (A)
  static const int tR011[nL011]={122,121,122,121,122,121};
  static const int tR012[nL012]={121,122,121,122,121,122};
  static const int tR013[nL013]={122,121,122,121,122,121};
  static const int tR014[nL014]={121,122,121,122,121,122,121}; //____________________12_
  static const int tR015[nL015]={122,121,242,241,242,241,242}; // (6)
  static const int tR016[nL016]={241,242,241,242,241,242,241};
  static const int tR017[nL017]={242,241,242,241,242,241,242};
  static const int tR018[nL018]={241,242,241,242,243,242,241};
  static const int tR019[nL019]={242,241,242,241,242,241,242};
  static const int tR020[nL020]={241,242,241,242,241,242,241,242};
  static const int tR021[nL021]={242,241,242,241,242,241,242,241}; // (5)
  static const int tR022[nL022]={241,242,241,242,241,242,241,242}; //________________24_
  static const int tR023[nL023]={232,231,232,231,232,231,232,231};
  static const int tR024[nL024]={231,232,231,232,231,232,231,232};
  static const int tR025[nL025]={232,231,232,231,232,231,232,231,232};
  static const int tR026[nL026]={231,232,231,232,233,232,231,232,231};
  static const int tR027[nL027]={232,231,232,231,232,231,232,231,232}; // (6)
  static const int tR028[nL028]={231,232,231,232,231,232,231,232,231};
  static const int tR029[nL029]={232,231,232,231,232,231,232,231,232};
  static const int tR030[nL030]={231,232,231,232,231,232,231,232,231};
  static const int tR031[nL031]={232,231,232,231,232,231,232,231,232,231}; //________23_
  static const int tR032[nL032]={231,232,231,222,221,222,221,222,221,222};
  static const int tR033[nL033]={222,221,222,221,222,221,222,221,222,221}; // (6)
  static const int tR034[nL034]={221,222,221,222,221,222,221,222,221,222};
  static const int tR035[nL035]={222,221,222,221,222,221,222,221,222,221};
  static const int tR036[nL036]={221,222,221,222,223,222,221,222,221,222};
  static const int tR037[nL037]={222,221,222,221,222,221,222,221,222,221,222};
  static const int tR038[nL038]={221,222,221,222,221,222,221,222,221,222,221};
  static const int tR039[nL039]={222,221,222,221,222,221,222,221,222,221,222}; // (5)
  static const int tR040[nL040]={221,222,221,222,221,222,221,222,221,222,221};//_____22_
  static const int tR041[nL041]={212,211,212,211,212,211,212,211,212,211,212};
  static const int tR042[nL042]={211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR043[nL043]={212,211,212,211,212,211,212,211,212,211,212,211};
  static const int tR044[nL044]={211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR045[nL045]={212,211,212,211,212,211,212,211,212,211,212,211};//(6)
  static const int tR046[nL046]={211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR047[nL047]={212,211,212,211,212,211,212,211,212,211,212,211};
  static const int tR048[nL048]={211,212,211,212,211,212,211,214,211,212,211,212,211};
  static const int tR049[nL049]={212,211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR050[nL050]={211,212,211,212,211,212,211,212,211,212,211,212,211};
  static const int tR051[nL051]={212,211,212,211,212,211,212,211,212,211,212,211,212};//(6)
  static const int tR052[nL052]={211,212,211,212,211,212,211,212,211,212,211,212,211};
  static const int tR053[nL053]={212,211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR054[nL054]={211,212,211,212,211,212,211,212,211,212,211,212,211,212};
  static const int tR055[nL055]={212,211,212,211,212,211,212,211,212,211,212,211,212,211};
  // _______________________________________________________________________________21_ (5)
  static const int tR056[nL056]={211,212,211,202,201,202,201,202,201,202,201,202,201,202};
  static const int tR057[nL057]={202,201,202,201,202,201,202,201,202,201,202,201,202,201};
  static const int tR058[nL058]={201,202,201,202,201,202,201,202,201,202,201,202,201,202};
  static const int tR059[nL059]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202};
  static const int tR060[nL060]={201,202,201,202,201,202,201,202,201,202,201,202,201,202,
                                 201};
  static const int tR061[nL061]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202}; // (6)
  static const int tR062[nL062]={201,202,201,202,201,202,201,204,201,202,201,202,201,202,
                                 201};
  static const int tR063[nL063]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202};
  static const int tR064[nL064]={201,202,201,202,201,202,201,202,201,202,201,202,201,202,
                                 201};
  static const int tR065[nL065]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202,201};
  static const int tR066[nL066]={201,202,201,202,201,202,201,202,201,202,201,202,201,202,
                                 201,202}; // (6)
  static const int tR067[nL067]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202,201};
  static const int tR068[nL068]={201,202,201,202,201,202,201,202,201,202,201,202,201,202,
                                 201,202};
  static const int tR069[nL069]={202,201,202,201,202,201,202,201,202,201,202,201,202,201,
                                 202,201};
  static const int tR070[nL070]={201,202,201,202,201,202,201,202,201,202,201,202,201,202,
                                 201,202};
  static const int tR071[nL071]={202,201,202,201,202,201,202,201,202,201,192,191,192,191,
                                 192,191,192}; // ___________________________________20_
  static const int tR072[nL072]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191};
  static const int tR073[nL073]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192}; // (5)
  static const int tR074[nL074]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191};
  static const int tR075[nL075]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192};
  static const int tR076[nL076]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191,192};
  static const int tR077[nL077]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191};
  static const int tR078[nL078]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191,192}; // (6)
  static const int tR079[nL079]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191};
  static const int tR080[nL080]={191,192,191,192,191,192,191,194,191,192,191,192,191,192,
                                 191,192,191,192};
  static const int tR081[nL081]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191};
  static const int tR082[nL082]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191,192,191};
  static const int tR083[nL083]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191,192};
  static const int tR084[nL084]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191,192,191}; // (6)
  static const int tR085[nL085]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191,192};
  static const int tR086[nL086]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,191,192,191};
  static const int tR087[nL087]={192,191,192,191,192,191,192,191,192,191,192,191,192,191,
                                 192,191,192,191,192};
  static const int tR088[nL088]={191,192,191,192,191,192,191,192,191,192,191,192,191,192,
                                 191,192,181,182,181,182}; // _______________________19_
  // ------------------------------------------------------------------------------------
  static const int tR089[nL089]={192,191,192,191,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181}; // (5)
  static const int tR090[nL090]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182};
  static const int tR091[nL091]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181};
  static const int tR092[nL092]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182};
  static const int tR093[nL093]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182};
  static const int tR094[nL094]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181};
  static const int tR095[nL095]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182}; // (6)
  static const int tR096[nL096]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181};
  static const int tR097[nL097]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182};
  static const int tR098[nL098]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181};
  static const int tR099[nL099]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181};
  static const int tR100[nL100]={181,182,181,182,181,182,181,182,181,182,181,182,183,182,
                                 181,182,181,182,181,182,181,182}; // (6)
  static const int tR101[nL101]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181};
  static const int tR102[nL102]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181,182};
  static const int tR103[nL103]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181};
  static const int tR104[nL104]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181,182};
  static const int tR105[nL105]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181,182};
  static const int tR106[nL106]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181,182,181};
  static const int tR107[nL107]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181,182}; // (5)
  static const int tR108[nL108]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181,182,181};
  static const int tR109[nL109]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,181,182,181,182,181,182};
  static const int tR110[nL110]={181,182,181,182,181,182,181,182,181,182,181,182,181,182,
                                 181,182,181,182,181,182,181,182,181,182};
  static const int tR111[nL111]={182,181,182,181,182,181,182,181,182,181,182,181,182,181,
                                 182,181,182,171,172,171,172,171,172,171}; // _________4_
  static const int tR112[nL112]={181,182,181,182,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172}; // (6)
  static const int tR113[nL113]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171};
  static const int tR114[nL114]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172};
  static const int tR115[nL115]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171};
  static const int tR116[nL116]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171};
  static const int tR117[nL117]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172};
  static const int tR118[nL118]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171};
  static const int tR119[nL119]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172}; // (6)
  static const int tR120[nL120]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171};
  static const int tR121[nL121]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172};
  static const int tR122[nL122]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR123[nL123]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR124[nL124]={171,172,171,172,171,172,171,172,171,172,171,172,173,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172};// (5)
  static const int tR125[nL125]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR126[nL126]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR127[nL127]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR128[nL128]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR129[nL129]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR130[nL130]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172,171};//(6)
  static const int tR131[nL131]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR132[nL132]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR133[nL133]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR134[nL134]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR135[nL135]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,172,171,172,171,172,171};
  static const int tR136[nL136]={171,172,171,172,171,172,171,172,171,172,171,172,171,172,
                                 171,172,171,172,171,172,171,172,171,172,171,172,171,172};
  static const int tR137[nL137]={172,171,172,171,172,171,172,171,172,171,172,171,172,171,
                                 172,171,172,171,172,171,172,171,162,161,162,161,162,161};
  // ____________________________________________________________________________(6)___3_
  static const int tR138[nL138]={171,172,171,172,171,172,171,172,171,172,171,172,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162};
  static const int tR139[nL139]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162};
  static const int tR140[nL140]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161};
  static const int tR141[nL141]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162}; // (5)
  static const int tR142[nL142]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161};
  static const int tR143[nL143]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162};
  static const int tR144[nL144]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162};
  static const int tR145[nL145]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161};
  static const int tR146[nL146]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,163,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162};
  static const int tR147[nL147]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161}; // (6)
  static const int tR148[nL148]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162};
  static const int tR149[nL149]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161};
  static const int tR150[nL150]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161};
  static const int tR151[nL151]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162};
  static const int tR152[nL152]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161}; // (6)
  static const int tR153[nL153]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162};
  static const int tR154[nL154]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161};
  static const int tR155[nL155]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162};
  static const int tR156[nL156]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162};
  static const int tR157[nL157]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161};
  static const int tR158[nL158]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162}; // (5)
  static const int tR159[nL159]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161};
  static const int tR160[nL160]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,163,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162};
  static const int tR161[nL161]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162};
  static const int tR162[nL162]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161};
  static const int tR163[nL163]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162}; // (6)
  static const int tR164[nL164]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161};
  static const int tR165[nL165]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162};
  static const int tR166[nL166]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161};
  static const int tR167[nL167]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,162,161};
  static const int tR168[nL168]={161,162,161,162,161,162,161,162,161,162,161,162,161,162,
                                 161,162,161,162,161,162,161,162,161,162,161,162,161,152,
                                 151,152,151,152,151,152}; // _________________________2_
  static const int tR169[nL169]={162,161,162,161,162,161,162,161,162,161,162,161,162,161,
                                 162,161,162,161,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151}; // (6)
  static const int tR170[nL170]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152};
  static const int tR171[nL171]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151};
  static const int tR172[nL172]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152};
  static const int tR173[nL173]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152};
  static const int tR174[nL174]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151};
  static const int tR175[nL175]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152}; // (5)
  static const int tR176[nL176]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151};
  static const int tR177[nL177]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152};
  static const int tR178[nL178]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,153,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152};
  static const int tR179[nL179]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151};
  static const int tR180[nL180]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152}; // (6)
  static const int tR181[nL181]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151};
  static const int tR182[nL182]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152};
  static const int tR183[nL183]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151};
  static const int tR184[nL184]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151};
  static const int tR185[nL185]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152};
  static const int tR186[nL186]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151};
  static const int tR187[nL187]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152}; // (6)
  static const int tR188[nL188]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151};
  static const int tR189[nL189]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152};
  static const int tR190[nL190]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152};
  static const int tR191[nL191]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151};
  static const int tR192[nL192]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152}; // (5)
  static const int tR193[nL193]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151};
  static const int tR194[nL194]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152};
  static const int tR195[nL195]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152};
  static const int tR196[nL196]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,153,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151};
  static const int tR197[nL197]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152}; // (6)
  static const int tR198[nL198]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151};
  static const int tR199[nL199]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152};
  static const int tR200[nL200]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151};
  static const int tR201[nL201]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151};
  static const int tR202[nL202]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152};
  static const int tR203[nL203]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151}; //(6)
  static const int tR204[nL204]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,151,152};
  static const int tR205[nL205]={152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,152,151,152,151,152,151,152,151,152,151,
                                 152,151,152,151,142,141,142,141,142,141,142,141};
  static const int tR206[nL206]={151,152,151,152,151,152,151,152,151,152,151,152,151,152,
                                 151,152,151,152,151,152,151,152,151,152,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142};//__1_
  static const int tR207[nL207]={152,151,152,151,152,151,152,151,152,151,152,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142};
  static const int tR208[nL208]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141};
  static const int tR209[nL209]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142};//(5)
  static const int tR210[nL210]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141};
  static const int tR211[nL211]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142};
  static const int tR212[nL212]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,143,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142};
  static const int tR213[nL213]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141};
  static const int tR214[nL214]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142};
  static const int tR215[nL215]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141};
  static const int tR216[nL216]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142};
  static const int tR217[nL217]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141};
  static const int tR218[nL218]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141};
  static const int tR219[nL219]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142};
  static const int tR220[nL220]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141};
  static const int tR221[nL221]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142}; // (6)
  static const int tR222[nL222]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141};
  static const int tR223[nL223]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142};
  static const int tR224[nL224]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142};
  static const int tR225[nL225]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141};
  static const int tR226[nL226]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,143,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142}; // (5)
  static const int tR227[nL227]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141};
  static const int tR228[nL228]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142};
  static const int tR229[nL229]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142};
  static const int tR230[nL230]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141};
  static const int tR231[nL231]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,142,141,142,  0,  0,  0,
                                   0,  0,  0}; // (5+1=6)
  static const int tR232[nL232]={141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,141,142,141,142,141,142,141,142,141,142,
                                 141,142,141,142,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0};
  static const int tR233[nL233]={142,141,142,141,142,141,142,141,142,141,142,141,142,141,
                                 142,141,142,141,142,141,142,141,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0};
  //------------------------------------------------------------------------------------
  static const int tL001[nL001]={131,132,131,132}; // Left Part of the widget (-phi)
  static const int tL002[nL002]={132,131,132,131};
  static const int tL003[nL003]={131,132,131,132,131};
  static const int tL004[nL004]={132,131,132,131,132}; // (5)
  static const int tL005[nL005]={131,132,131,132,131};
  static const int tL006[nL006]={132,131,132,131,132};
  static const int tL007[nL007]={131,132,131,132,131};
  static const int tL008[nL008]={132,131,132,131,132,131}; // ______________________13_
  static const int tL009[nL009]={121,122,121,122,121,122};
  static const int tL010[nL010]={122,121,122,121,124,121};
  static const int tL011[nL011]={121,122,121,122,121,122}; // (6) (B)
  static const int tL012[nL012]={122,121,122,121,122,121};
  static const int tL013[nL013]={121,122,121,122,121,122};
  static const int tL014[nL014]={122,121,122,121,122,121,122}; //___________________12_
  static const int tL015[nL015]={121,122,111,112,111,112,111};
  static const int tL016[nL016]={112,111,112,111,112,111,112};
  static const int tL017[nL017]={111,112,111,112,111,112,111}; // (6)
  static const int tL018[nL018]={112,111,112,111,114,111,112};
  static const int tL019[nL019]={111,112,111,112,111,112,111};
  static const int tL020[nL020]={112,111,112,111,112,111,112,111};
  static const int tL021[nL021]={111,112,111,112,111,112,111,112}; // (5)
  static const int tL022[nL022]={112,111,112,111,112,111,112,111}; //_______________11_
  static const int tL023[nL023]={101,102,101,102,101,102,101,102};
  static const int tL024[nL024]={102,101,102,101,102,101,102,101};
  static const int tL025[nL025]={101,102,101,102,101,102,101,102,101};
  static const int tL026[nL026]={102,101,102,101,104,101,102,101,102};
  static const int tL027[nL027]={101,102,101,102,101,102,101,102,101}; // (6)
  static const int tL028[nL028]={102,101,102,101,102,101,102,101,102};
  static const int tL029[nL029]={101,102,101,102,101,102,101,102,101};
  static const int tL030[nL030]={102,101,102,101,102,101,102,101,102};
  static const int tL031[nL031]={101,102,101,102,101,102,101,102,101,102}; //_______10_
  static const int tL032[nL032]={102,101,102, 91, 92, 91, 92, 91, 92, 91};
  static const int tL033[nL033]={ 91, 92, 91, 92, 91, 92, 91, 92, 91, 92}; // (6)
  static const int tL034[nL034]={ 92, 91, 92, 91, 92, 91, 92, 91, 92, 91};
  static const int tL035[nL035]={ 91, 92, 91, 92, 91, 92, 91, 92, 91, 92};
  static const int tL036[nL036]={ 92, 91, 92, 91, 94, 91, 92, 91, 92, 91};
  static const int tL037[nL037]={ 91, 92, 91, 92, 91, 92, 91, 92, 91, 92, 91};
  static const int tL038[nL038]={ 92, 91, 92, 91, 92, 91, 92, 91, 92, 91, 92};
  static const int tL039[nL039]={ 91, 92, 91, 92, 91, 92, 91, 92, 91, 92, 91}; // (5)
  static const int tL040[nL040]={ 92, 91, 92, 91, 92, 91, 92, 91, 92, 91, 92};
  static const int tL041[nL041]={ 91, 92, 91, 92, 91, 92, 91, 92, 91, 92, 91};
  static const int tL042[nL042]={ 92, 91, 92, 91, 92, 91, 92, 91, 92, 91, 92, 91};//_9_
  static const int tL043[nL043]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};
  static const int tL044[nL044]={ 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL045[nL045]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};//(6)
  static const int tL046[nL046]={ 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL047[nL047]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};
  static const int tL048[nL048]={ 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};
  static const int tL049[nL049]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL050[nL050]={ 82, 81, 82, 81, 82, 81, 82, 81, 84, 81, 82, 81, 82};//(6)
  static const int tL051[nL051]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL052[nL052]={ 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};
  static const int tL053[nL053]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL054[nL054]={ 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81};
  static const int tL055[nL055]={ 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82, 81, 82};
  // ________________________________________________________________________________8_ (5)
  static const int tL056[nL056]={ 82, 81, 82, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71};
  static const int tL057[nL057]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72};
  static const int tL058[nL058]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71};
  static const int tL059[nL059]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71};
  static const int tL060[nL060]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71,
                                  72};
  static const int tL061[nL061]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71}; // (6)
  static const int tL062[nL062]={ 72, 71, 72, 71, 72, 71, 72, 71, 74, 71, 72, 71, 72, 71,
                                  71};
  static const int tL063[nL063]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71};
  static const int tL064[nL064]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71,
                                  72};
  static const int tL065[nL065]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71, 72};
  static const int tL066[nL066]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71,
                                  72, 71}; // (6)
  static const int tL067[nL067]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71, 72};
  static const int tL068[nL068]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71,
                                  72, 71};
  static const int tL069[nL069]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72,
                                  71, 72};
  static const int tL070[nL070]={ 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 71,
                                  72, 71};
  static const int tL071[nL071]={ 71, 72, 71, 72, 71, 72, 71, 72, 71, 72, 61, 62, 61, 62,
                                  61, 62, 61}; // _____________________________________7_
  static const int tL072[nL072]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62};
  static const int tL073[nL073]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61}; // (5)
  static const int tL074[nL074]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62};
  static const int tL075[nL075]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61};
  static const int tL076[nL076]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61};
  static const int tL077[nL077]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62};
  static const int tL078[nL078]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61}; // (6)
  static const int tL079[nL079]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62};
  static const int tL080[nL080]={ 62, 61, 62, 61, 62, 61, 62, 61, 64, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61};
  static const int tL081[nL081]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62};
  static const int tL082[nL082]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61, 62};
  static const int tL083[nL083]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62, 61}; // (6)
  static const int tL084[nL084]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61, 62};
  static const int tL085[nL085]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62, 61};
  static const int tL086[nL086]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 62, 61, 62};
  static const int tL087[nL087]={ 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62,
                                  61, 62, 61, 62, 61};
  static const int tL088[nL088]={ 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61, 62, 61,
                                  62, 61, 52, 51, 52, 51}; // _________________________6_
  //-------------------------------------------------------------------------------------
  static const int tL089[nL089]={ 61, 62, 61, 62, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52}; // (5)
  static const int tL090[nL090]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51};
  static const int tL091[nL091]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52};
  static const int tL092[nL092]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51};
  static const int tL093[nL093]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51};
  static const int tL094[nL094]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52};
  static const int tL095[nL095]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51}; // (6)
  static const int tL096[nL096]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52};
  static const int tL097[nL097]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51};
  static const int tL098[nL098]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52};
  static const int tL099[nL099]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52};
  static const int tL100[nL100]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 53, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51}; // (6)
  static const int tL101[nL101]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52};
  static const int tL102[nL102]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51};
  static const int tL103[nL103]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52};
  static const int tL104[nL104]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51};
  static const int tL105[nL105]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52, 51};
  static const int tL106[nL106]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51, 52};
  static const int tL107[nL107]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52, 51}; // (5)
  static const int tL108[nL108]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51, 52};
  static const int tL109[nL109]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 52, 51, 52, 51, 52, 51};
  static const int tL110[nL110]={ 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51,
                                  52, 51, 52, 51, 52, 51, 52, 51, 52, 51};
  static const int tL111[nL111]={ 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52, 51, 52,
                                  51, 52, 51, 42, 41, 42, 41, 42, 41, 42}; // _________4_
  static const int tL112[nL112]={ 52, 51, 52, 51, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41}; // (6)
  static const int tL113[nL113]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL114[nL114]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL115[nL115]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL116[nL116]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL117[nL117]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL118[nL118]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL119[nL119]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41}; // (6)
  static const int tL120[nL120]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL121[nL121]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL122[nL122]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL123[nL123]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL124[nL124]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 43, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};// (5)
  static const int tL125[nL125]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL126[nL126]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL127[nL127]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL128[nL128]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL129[nL129]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL130[nL130]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};//(6)
  static const int tL131[nL131]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL132[nL132]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL133[nL133]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL134[nL134]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL135[nL135]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42};
  static const int tL136[nL136]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41,
                                  42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41};
  static const int tL137[nL137]={ 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42,
                                  41, 42, 41, 42, 41, 42, 41, 42, 31, 32, 31, 32, 31, 32};
  // ____________________________________________________________________________(6)___3_
  static const int tL138[nL138]={ 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 42, 41, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31};
  static const int tL139[nL139]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31};
  static const int tL140[nL140]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32};
  static const int tL141[nL141]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31}; // (5)
  static const int tL142[nL142]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32};
  static const int tL143[nL143]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31};
  static const int tL144[nL144]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31};
  static const int tL145[nL145]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32};
  static const int tL146[nL146]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 33, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31};
  static const int tL147[nL147]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32}; // (6)
  static const int tL148[nL148]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31};
  static const int tL149[nL149]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32};
  static const int tL150[nL150]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32};
  static const int tL151[nL151]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31};
  static const int tL152[nL152]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32}; // (6)
  static const int tL153[nL153]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31};
  static const int tL154[nL154]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32};
  static const int tL155[nL155]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31};
  static const int tL156[nL156]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31};
  static const int tL157[nL157]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32};
  static const int tL158[nL158]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31}; // (5)
  static const int tL159[nL159]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32};
  static const int tL160[nL160]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 33, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31};
  static const int tL161[nL161]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31};
  static const int tL162[nL162]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32};
  static const int tL163[nL163]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31}; // (6)
  static const int tL164[nL164]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32};
  static const int tL165[nL165]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31};
  static const int tL166[nL166]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32};
  static const int tL167[nL167]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 31, 32};
  static const int tL168[nL168]={ 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31,
                                  32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 21,
                                  22, 21, 22, 21, 22, 21}; // _________________________2_
  static const int tL169[nL169]={ 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32, 31, 32,
                                  31, 32, 31, 32, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22}; // (6)
  static const int tL170[nL170]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21};
  static const int tL171[nL171]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22};
  static const int tL172[nL172]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21};
  static const int tL173[nL173]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21};
  static const int tL174[nL174]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22};
  static const int tL175[nL175]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21}; // (5)
  static const int tL176[nL176]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22};
  static const int tL177[nL177]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21};
  static const int tL178[nL178]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 23, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL179[nL179]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL180[nL180]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21}; // (6)
  static const int tL181[nL181]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL182[nL182]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL183[nL183]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL184[nL184]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL185[nL185]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL186[nL186]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL187[nL187]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21}; // (6)
  static const int tL188[nL188]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL189[nL189]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL190[nL190]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL191[nL191]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL192[nL192]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21}; // (5)
  static const int tL193[nL193]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL194[nL194]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL195[nL195]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL196[nL196]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 23, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL197[nL197]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21}; // (6)
  static const int tL198[nL198]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL199[nL199]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL200[nL200]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL201[nL201]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22};
  static const int tL202[nL202]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL203[nL203]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22}; //(6)
  static const int tL204[nL204]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21};
  static const int tL205[nL205]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22,
                                  21, 22, 21, 22, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL206[nL206]={ 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21,
                                  22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};//__1_
  static const int tL207[nL207]={ 21, 22, 21, 22, 21, 22, 21, 22, 21, 22, 21, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};
  static const int tL208[nL208]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL209[nL209]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};//(5)
  static const int tL210[nL210]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL211[nL211]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};
  static const int tL212[nL212]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 13, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};
  static const int tL213[nL213]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL214[nL214]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};
  static const int tL215[nL215]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL216[nL216]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11};
  static const int tL217[nL217]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12};
  static const int tL218[nL218]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12};
  static const int tL219[nL219]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11};
  static const int tL220[nL220]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12};
  static const int tL221[nL221]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11}; // (6)
  static const int tL222[nL222]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12};
  static const int tL223[nL223]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11};
  static const int tL224[nL224]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11};
  static const int tL225[nL225]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12};
  static const int tL226[nL226]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 13, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11}; // (5)
  static const int tL227[nL227]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12};
  static const int tL228[nL228]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11};
  static const int tL229[nL229]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11};
  static const int tL230[nL230]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12};
  static const int tL231[nL231]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,  0,  0,  0,
                                   0,  0,  0}; // (5+1=6)
  static const int tL232[nL232]={ 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
                                  12, 11, 12, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0};
  static const int tL233[nL233]={ 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
                                  11, 12, 11, 12, 11, 12, 11, 12,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                                   0,  0,  0};
  static const int nSL[nLay]={
    nL001, nL002, nL003, nL004, nL005, nL006, nL007, nL008, nL009 ,nL010,
    nL011, nL012, nL013, nL014, nL015, nL016, nL017, nL018, nL019 ,nL020,
    nL021, nL022, nL023, nL024, nL025, nL026, nL027, nL028, nL029 ,nL030,
    nL031, nL032, nL033, nL034, nL035, nL036, nL037, nL038, nL039 ,nL040,
    nL041, nL042, nL043, nL044, nL045, nL046, nL047, nL048, nL049 ,nL050,
    nL051, nL052, nL053, nL054, nL055, nL056, nL057, nL058, nL059 ,nL060,
    nL061, nL062, nL063, nL064, nL065, nL066, nL067, nL068, nL069 ,nL070,
    nL071, nL072, nL073, nL074, nL075, nL076, nL077, nL078, nL079 ,nL080,
    nL081, nL082, nL083, nL084, nL085, nL086, nL087, nL088, nL089 ,nL090,
    nL091, nL092, nL093, nL094, nL095, nL096, nL097, nL098, nL099 ,nL100,
    nL101, nL102, nL103, nL104, nL105, nL106, nL107, nL108, nL109 ,nL110,
    nL111, nL112, nL113, nL114, nL115, nL116, nL117, nL118, nL119 ,nL120,
    nL121, nL122, nL123, nL124, nL125, nL126, nL127, nL128, nL129 ,nL130,
    nL131, nL132, nL133, nL134, nL135, nL136, nL137, nL138, nL139 ,nL140,
    nL141, nL142, nL143, nL144, nL145, nL146, nL147, nL148, nL149 ,nL150,
    nL151, nL152, nL153, nL154, nL155, nL156, nL157, nL158, nL159 ,nL160,
    nL161, nL162, nL163, nL164, nL165, nL166, nL167, nL168, nL169 ,nL170,
    nL171, nL172, nL173, nL174, nL175, nL176, nL177, nL178, nL179 ,nL180,
    nL181, nL182, nL183, nL184, nL185, nL186, nL187, nL188, nL189 ,nL190,
    nL191, nL192, nL193, nL194, nL195, nL196, nL197, nL198, nL199 ,nL200,
    nL201, nL202, nL203, nL204, nL205, nL206, nL207, nL208, nL209 ,nL210,
    nL211, nL212, nL213, nL214, nL215, nL216, nL217, nL218, nL219 ,nL220,
    nL221, nL222, nL223, nL224, nL225, nL226, nL227, nL228, nL229 ,nL230,
    nL231, nL232, nL233};
  static const int * const nLT[nLay]={
    tL001, tL002, tL003, tL004, tL005, tL006, tL007, tL008, tL009 ,tL010,
    tL011, tL012, tL013, tL014, tL015, tL016, tL017, tL018, tL019 ,tL020,
    tL021, tL022, tL023, tL024, tL025, tL026, tL027, tL028, tL029 ,tL030,
    tL031, tL032, tL033, tL034, tL035, tL036, tL037, tL038, tL039 ,tL040,
    tL041, tL042, tL043, tL044, tL045, tL046, tL047, tL048, tL049 ,tL050,
    tL051, tL052, tL053, tL054, tL055, tL056, tL057, tL058, tL059 ,tL060,
    tL061, tL062, tL063, tL064, tL065, tL066, tL067, tL068, tL069 ,tL070,
    tL071, tL072, tL073, tL074, tL075, tL076, tL077, tL078, tL079 ,tL080,
    tL081, tL082, tL083, tL084, tL085, tL086, tL087, tL088, tL089 ,tL090,
    tL091, tL092, tL093, tL094, tL095, tL096, tL097, tL098, tL099 ,tL100,
    tL101, tL102, tL103, tL104, tL105, tL106, tL107, tL108, tL109 ,tL110,
    tL111, tL112, tL113, tL114, tL115, tL116, tL117, tL118, tL119 ,tL120,
    tL121, tL122, tL123, tL124, tL125, tL126, tL127, tL128, tL129 ,tL130,
    tL131, tL132, tL133, tL134, tL135, tL136, tL137, tL138, tL139 ,tL140,
    tL141, tL142, tL143, tL144, tL145, tL146, tL147, tL148, tL149 ,tL150,
    tL151, tL152, tL153, tL154, tL155, tL156, tL157, tL158, tL159 ,tL160,
    tL161, tL162, tL163, tL164, tL165, tL166, tL167, tL168, tL169 ,tL170,
    tL171, tL172, tL173, tL174, tL175, tL176, tL177, tL178, tL179 ,tL180,
    tL181, tL182, tL183, tL184, tL185, tL186, tL187, tL188, tL189 ,tL190,
    tL191, tL192, tL193, tL194, tL195, tL196, tL197, tL198, tL199 ,tL200,
    tL201, tL202, tL203, tL204, tL205, tL206, tL207, tL208, tL209 ,tL210,
    tL211, tL212, tL213, tL214, tL215, tL216, tL217, tL218, tL219 ,tL220,
    tL221, tL222, tL223, tL224, tL225, tL226, tL227, tL228, tL229 ,tL230,
    tL231, tL232, tL233};
  static const int * const nRT[nLay]={
    tR001, tR002, tR003, tR004, tR005, tR006, tR007, tR008, tR009 ,tR010,
    tR011, tR012, tR013, tR014, tR015, tR016, tR017, tR018, tR019 ,tR020,
    tR021, tR022, tR023, tR024, tR025, tR026, tR027, tR028, tR029 ,tR030,
    tR031, tR032, tR033, tR034, tR035, tR036, tR037, tR038, tR039 ,tR040,
    tR041, tR042, tR043, tR044, tR045, tR046, tR047, tR048, tR049 ,tR050,
    tR051, tR052, tR053, tR054, tR055, tR056, tR057, tR058, tR059 ,tR060,
    tR061, tR062, tR063, tR064, tR065, tR066, tR067, tR068, tR069 ,tR070,
    tR071, tR072, tR073, tR074, tR075, tR076, tR077, tR078, tR079 ,tR080,
    tR081, tR082, tR083, tR084, tR085, tR086, tR087, tR088, tR089 ,tR090,
    tR091, tR092, tR093, tR094, tR095, tR096, tR097, tR098, tR099 ,tR100,
    tR101, tR102, tR103, tR104, tR105, tR106, tR107, tR108, tR109 ,tR110,
    tR111, tR112, tR113, tR114, tR115, tR116, tR117, tR118, tR119 ,tR120,
    tR121, tR122, tR123, tR124, tR125, tR126, tR127, tR128, tR129 ,tR130,
    tR131, tR132, tR133, tR134, tR135, tR136, tR137, tR138, tR139 ,tR140,
    tR141, tR142, tR143, tR144, tR145, tR146, tR147, tR148, tR149 ,tR150,
    tR151, tR152, tR153, tR154, tR155, tR156, tR157, tR158, tR159 ,tR160,
    tR161, tR162, tR163, tR164, tR165, tR166, tR167, tR168, tR169 ,tR170,
    tR171, tR172, tR173, tR174, tR175, tR176, tR177, tR178, tR179 ,tR180,
    tR181, tR182, tR183, tR184, tR185, tR186, tR187, tR188, tR189 ,tR190,
    tR191, tR192, tR193, tR194, tR195, tR196, tR197, tR198, tR199 ,tR200,
    tR201, tR202, tR203, tR204, tR205, tR206, tR207, tR208, tR209 ,tR210,
    tR211, tR212, tR213, tR214, tR215, tR216, tR217, tR218, tR219 ,tR220,
    tR221, tR222, tR223, tR224, tR225, tR226, tR227, tR228, tR229 ,tR230,
    tR231, tR232, tR233};

  /*
  // The following are differences in the Source tube positions(not used so far)
  // *** At present for all widgets the F01 is used (@@ to be developed M.K.)
  static const int nS=31;      // a # of the source tubes in the widget 
  // 0 - H(Long), 1 - E(Short)
  //                                                1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2
  //                        1 1 2 2 3 3 4 5 6 7 8 9 0 1 2 2 3 4 4 5 5 6 6 7 8 9 0 1 2 3 4
  //                        A B A B A B                 A B   A B A B A B
  static const int F01[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F02[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F03[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,0};
  static const int F04[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F05[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F06[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F07[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F08[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F09[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F10[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F11[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F12[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F13[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F14[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F15[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};
  static const int F16[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int F17[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int F18[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0};

  static const int B01[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B02[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B03[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B04[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B05[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B06[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B07[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B08[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B09[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B10[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B11[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B12[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B13[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B14[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  static const int B15[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B16[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B17[nS]={0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  static const int B18[nS]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0};
  */

  static const double cellSize = 0.5*CLHEP::cm;  // 0.5 cm is the cell size
  if (!(xl > 0.))
    xl=-xl;
  double     fx=xl/cellSize;
  int ny=static_cast<int>((yl-yMin)/cellSize);   // Layer number (starting from 0)
  if (ny < 0 || ny >= nLay) {// Sould never happen as was checked beforehand
    LogDebug("HFShower") << "-Warning-HFFibreFiducial::PMTNumber: "
				 << "check limits y = " << yl << ", nL=" 
				 << nLay;
    return 0;
  }
  int nx=static_cast<int>(fx);           // Cell number (starting from 0)
  LogDebug("HFShower") << "HFFibreFiducial::PMTNumber:X = " << xv 
			       << ", Y = " << yv << ", Z = " << pe_effect.z()
			       << ", fX = " << fx << "-> nX = " << nx 
			       << ", nY = " << ny << ", mX = " << nSL[ny]
			       << ", x = " << xl << ", y = "<< yl << ", s = "
			       << cellSize << ", nW = " << nwid << ", phi = " 
			       << phi/CLHEP::deg << ", phis = " 
			       << atan2(xl, yl)/CLHEP::deg
			       << ", phir = " << phir/CLHEP::deg;
  if (nx >= nSL[ny]) {
    LogDebug("HFShower") << "HFFibreFiducial::nx/ny (" << nx 
				 << "," << ny <<") " << " above limit " 
				 << nSL[ny];
    return 0;                           // ===> out of the acceptance
  }
  int code=0;                           // a prototype
  if (left) code=nLT[ny][nx];
  else      code=nRT[ny][nx];
  int flag= code%10;
  int npmt= code/10;
  bool src= false;                       // by default: not a source-tube
  LogDebug("HFShower") << "HFFibreFiducial::nx/ny (" << nx << ","
			       << ny << ") code/flag/npmt " <<  code << "/" 
			       << flag << "/" << npmt;

  if (!flag) return 0;                   // ===> no fiber in the cell
  else if (flag==1) npmt += 24;
  else if (flag==3 || flag==4) {
    src=true;
  }
  LogDebug("HFShower") << "HFFibreFiducial::PMTNumber: src = " << src 
			       << ", npmt =" << npmt;
  if (src) return -npmt;   // return the negative number for the source
  return npmt;
} // End of PMTNumber
