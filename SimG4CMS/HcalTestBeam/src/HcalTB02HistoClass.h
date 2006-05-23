#ifndef HcalTestBeam_HcalTB02HistoClass_H
#define HcalTestBeam_HcalTB02HistoClass_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02HistoClass
//
/**\class HcalTB02HistoClass HcalTB02HistoClass.h SimG4CMS/HcalTestBeam/interface/HcalTB02HistoClass.h
  
 Description: Content of the Root tree for Hcal Test Beam 2002 studies
  
 Usage: Used in 2002 Hcal Test Beam studies
*/
//
// Original Author:  
//         Created:  Thu Sun 21 10:14:34 CEST 2006
// $Id$
//
  
// system include files
 
// user include files

class HcalTB02HistoClass {
   
public:
 
  // ---------- Constructor and destructor -----------------
  HcalTB02HistoClass() : Nprimaries(0), partID(0), Einit(0), eta(0), phi(0),
    Eentry(0), EhcalTot(0), Ehcal7x7(0), Ehcal5x5(0), EhcalTotN(0),
    Ehcal7x7N(0), Ehcal5x5N(0), Nunit(0), Ntimesli(0), xEentry(0),
    xEhcalTot(0), xEhcalTotN(0), xEhcal7x7(0), xEhcal5x5(0), xEhcal3x3(0),
    xEhcal7x7N(0), xEhcal5x5N(0), xEhcal3x3N(0), xNunit(0) {;}
  virtual ~HcalTB02HistoClass() {;}

  // ---------- member functions ---------------------------
  void set_partType(float v)  {partID = v;}
  void set_Nprim(float v)     {Nprimaries = v;}
  void set_Einit(float v)     {Einit = v;}
  void set_eta(float v)       {eta = v;}
  void set_phi(float v)       {phi = v;}
  void set_Eentry(float v)    {Eentry = v;}
  void set_ETot(float v)      {EhcalTot = v;}
  void set_E5x5(float v)      {Ehcal5x5 = v;}
  void set_E7x7(float v)      {Ehcal7x7 = v;}
  void set_ETotN(float v)     {EhcalTotN = v;}
  void set_E5x5N(float v)     {Ehcal5x5N = v;}
  void set_E7x7N(float v)     {Ehcal7x7N = v;}
  void set_NUnit(float v)     {Nunit = v;}
  void set_Ntimesli(float v)  {Ntimesli = v;}
  void set_xEentry(float v)   {xEentry = v;}
  void set_xETot(float v)     {xEhcalTot = v;}
  void set_xETotN(float v)    {xEhcalTotN = v;}
  void set_xE5x5(float v)     {xEhcal5x5 = v;}
  void set_xE3x3(float v)     {xEhcal3x3 = v;}
  void set_xE5x5N(float v)    {xEhcal5x5N = v;}
  void set_xE3x3N(float v)    {xEhcal3x3N = v;}
  void set_xNUnit(float v)    {xNunit = v;}
                                                                               
private:

  // ---------- Private Data members -----------------------
  float         Nprimaries;
  float         partID;
  float         Einit;
  float         eta;
  float         phi;
  float         Eentry;
  float         EhcalTot;
  float         Ehcal7x7;
  float         Ehcal5x5;
  float         EhcalTotN;
  float         Ehcal7x7N;
  float         Ehcal5x5N;
  float         Nunit;
  float         Ntimesli;
  float         xEentry;
  float         xEhcalTot;
  float         xEhcalTotN;
  float         xEhcal7x7;
  float         xEhcal5x5;
  float         xEhcal3x3;
  float         xEhcal7x7N;
  float         xEhcal5x5N;
  float         xEhcal3x3N;
  float         xNunit;
};
 
#endif
