#ifndef SimDataFormats_HcalTestBeam_HcalTB02HistoClass_H
#define SimDataFormats_HcalTestBeam_HcalTB02HistoClass_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HistoClass
//
/**\class HcalTB02HistoClass HcalTB02HistoClass.h SimDataFormats/HcalTestBeam/interface/HcalTB02HistoClass.h
  
 Description: Content of the Root tree for Hcal Test Beam 2002 studies
  
 Usage: Used in 2002 Hcal Test Beam studies
*/
//
// Original Author:  
//         Created:  Thu Sun 21 10:14:34 CEST 2006
// $Id: HcalTB02HistoClass.h,v 1.1 2006/11/13 10:04:36 sunanda Exp $
//
  
// system include files
 
// user include files

class HcalTB02HistoClass {
   
public:
 
  // ---------- Constructor and destructor -----------------
  HcalTB02HistoClass() : Nprimaries(0), partID_(0), Einit(0), eta_(0), phi_(0),
    Eentry(0), EhcalTot(0), Ehcal7x7(0), Ehcal5x5(0), EhcalTotN(0),
    Ehcal7x7N(0), Ehcal5x5N(0), Nunit(0), Ntimesli(0), xEentry_(0),
    xEhcalTot_(0), xEhcalTotN_(0), xEhcal7x7_(0), xEhcal5x5_(0), xEhcal3x3_(0),
    xEhcal7x7N_(0), xEhcal5x5N_(0), xEhcal3x3N_(0), xNunit_(0) {;}
  virtual ~HcalTB02HistoClass() {;}

  // ---------- member functions ---------------------------
  void set_partType(float v)  {partID_ = v;}
  void set_Nprim(float v)     {Nprimaries = v;}
  void set_Einit(float v)     {Einit = v;}
  void set_eta(float v)       {eta_ = v;}
  void set_phi(float v)       {phi_ = v;}
  void set_Eentry(float v)    {Eentry = v;}
  void set_ETot(float v)      {EhcalTot = v;}
  void set_E5x5(float v)      {Ehcal5x5 = v;}
  void set_E7x7(float v)      {Ehcal7x7 = v;}
  void set_ETotN(float v)     {EhcalTotN = v;}
  void set_E5x5N(float v)     {Ehcal5x5N = v;}
  void set_E7x7N(float v)     {Ehcal7x7N = v;}
  void set_NUnit(float v)     {Nunit = v;}
  void set_Ntimesli(float v)  {Ntimesli = v;}
  void set_xEentry(float v)   {xEentry_ = v;}
  void set_xETot(float v)     {xEhcalTot_ = v;}
  void set_xETotN(float v)    {xEhcalTotN_ = v;}
  void set_xE5x5(float v)     {xEhcal5x5_ = v;}
  void set_xE3x3(float v)     {xEhcal3x3_ = v;}
  void set_xE5x5N(float v)    {xEhcal5x5N_ = v;}
  void set_xE3x3N(float v)    {xEhcal3x3N_ = v;}
  void set_xNUnit(float v)    {xNunit_ = v;}

  float nPrimaries()  const {return Nprimaries;}
  float partID()      const {return partID_;}
  float eInit()       const {return Einit;}
  float eta()         const {return eta_;}
  float phi()         const {return phi_;}
  float eEntry()      const {return Eentry;}
  float eHcalTot()    const {return EhcalTot;}
  float eHcal7x7()    const {return Ehcal7x7;}
  float eHcal5x5()    const {return Ehcal5x5;}
  float eHcalTotN()   const {return EhcalTotN;}
  float eHcal7x7N()   const {return Ehcal7x7N;}
  float eHcal5x5N()   const {return Ehcal5x5N;}
  float nUnit()       const {return Nunit;}
  float nTimeSli()    const {return Ntimesli;}
  float xEntry()      const {return xEentry_;}
  float xEHcalTot()   const {return xEhcalTot_;}
  float xEHcalTotN()  const {return xEhcalTotN_;}
  float xEHcal7x7()   const {return xEhcal7x7_;}
  float xEHcal5x5()   const {return xEhcal5x5_;}
  float xEHcal3x3()   const {return xEhcal3x3_;}
  float xEHcal7x7N()  const {return xEhcal7x7N_;}
  float xEHcal5x5N()  const {return xEhcal5x5N_;}
  float xEHcal3x3N()  const {return xEhcal3x3N_;}
  float xNUnit()      const {return xNunit_;}
                                                                              
private:

  // ---------- Private Data members -----------------------
  float         Nprimaries;
  float         partID_;
  float         Einit;
  float         eta_;
  float         phi_;
  float         Eentry;
  float         EhcalTot;
  float         Ehcal7x7;
  float         Ehcal5x5;
  float         EhcalTotN;
  float         Ehcal7x7N;
  float         Ehcal5x5N;
  float         Nunit;
  float         Ntimesli;
  float         xEentry_;
  float         xEhcalTot_;
  float         xEhcalTotN_;
  float         xEhcal7x7_;
  float         xEhcal5x5_;
  float         xEhcal3x3_;
  float         xEhcal7x7N_;
  float         xEhcal5x5N_;
  float         xEhcal3x3N_;
  float         xNunit_;
};
 
#endif
