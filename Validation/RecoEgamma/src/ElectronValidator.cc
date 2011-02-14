
#include "Validation/RecoEgamma/interface/ElectronValidator.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <iostream>

ElectronValidator::ElectronValidator( const edm::ParameterSet& conf )
 {}

ElectronValidator::~ElectronValidator()
 {}

void ElectronValidator::prepareStore()
 {
  store_ = edm::Service<DQMStore>().operator->() ;
  if (!store_)
   { edm::LogError("ElectronValidator::prepareStore")<<"No DQMStore found !" ; }
 }

void ElectronValidator::setStoreFolder( const std::string & path )
 { store_->setCurrentFolder(path) ; }

void ElectronValidator::saveStore( const std::string & filename )
 { store_->save(filename) ; }

MonitorElement * ElectronValidator::bookH1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY )
 {
  MonitorElement * me = store_->book1D(name,title,nchX,lowX,highX) ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  return me ;
 }

MonitorElement * ElectronValidator::bookH1withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   const std::string & titleX, const std::string & titleY )
 {
  MonitorElement * me = store_->book1D(name,title,nchX,lowX,highX) ;
  me->getTH1F()->Sumw2() ;
  if (titleX!="") { me->getTH1F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH1F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  return me ;
 }

MonitorElement * ElectronValidator::bookH2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY )
 {
  MonitorElement * me = store_->book2D(name,title,nchX,lowX,highX,nchY,lowY,highY) ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  return me ;
 }

MonitorElement * ElectronValidator::bookH2withSumw2
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
   int nchY, double lowY, double highY,
   const std::string & titleX, const std::string & titleY )
 {
  MonitorElement * me = store_->book2D(name,title,nchX,lowX,highX,nchY,lowY,highY) ;
  me->getTH2F()->Sumw2() ;
  if (titleX!="") { me->getTH2F()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTH2F()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  return me ;
 }

MonitorElement * ElectronValidator::bookP1
 ( const std::string & name, const std::string & title,
   int nchX, double lowX, double highX,
             double lowY, double highY,
   const std::string & titleX, const std::string & titleY )
 {
  MonitorElement * me = store_->bookProfile(name,title,nchX,lowX,highX,lowY,highY) ;
  if (titleX!="") { me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { me->getTProfile()->GetYaxis()->SetTitle(titleY.c_str()) ; }
  return me ;
 }

MonitorElement * ElectronValidator::bookH1andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title, bool print )
 {
  TH1F * h_temp = (TH1F *)num->getTH1F()->Clone(name.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (print) { h_temp->Print() ; }
  MonitorElement * me = store_->book1D(name,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronValidator::bookH2andDivide
 ( const std::string & name, MonitorElement * num, MonitorElement * denom,
   const std::string & titleX, const std::string & titleY,
   const std::string & title, bool print )
 {
  TH2F * h_temp = (TH2F *)num->getTH2F()->Clone(name.c_str()) ;
  h_temp->Reset() ;
  h_temp->Divide(num->getTH1(),denom->getTH1(),1,1,"b") ;
  h_temp->GetXaxis()->SetTitle(titleX.c_str()) ;
  h_temp->GetYaxis()->SetTitle(titleY.c_str()) ;
  if (title!="") { h_temp->SetTitle(title.c_str()) ; }
  if (print) { h_temp->Print() ; }
  MonitorElement * me = store_->book2D(name,h_temp) ;
  delete h_temp ;
  return me ;
 }

MonitorElement * ElectronValidator::profileX
 ( const std::string & name, MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  if (me2d->getTH2F()->GetSumw2N()==0) me2d->getTH2F()->Sumw2() ; // workaround for http://savannah.cern.ch/bugs/?77751
  TProfile * p1_temp = me2d->getTH2F()->ProfileX() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = store_->bookProfile(name,p1_temp) ;
  delete p1_temp ;
  return me ;
 }

MonitorElement * ElectronValidator::profileY
 ( const std::string & name, MonitorElement * me2d,
   const std::string & title, const std::string & titleX, const std::string & titleY,
   Double_t minimum, Double_t maximum )
 {
  if (me2d->getTH2F()->GetSumw2N()==0) me2d->getTH2F()->Sumw2() ; // workaround for http://savannah.cern.ch/bugs/?77751
  TProfile * p1_temp = me2d->getTH2F()->ProfileY() ;
  if (title!="") { p1_temp->SetTitle(title.c_str()) ; }
  if (titleX!="") { p1_temp->GetXaxis()->SetTitle(titleX.c_str()) ; }
  if (titleY!="") { p1_temp->GetYaxis()->SetTitle(titleY.c_str()) ; }
  if (minimum!=-1111) { p1_temp->SetMinimum(minimum) ; }
  if (maximum!=-1111) { p1_temp->SetMaximum(maximum) ; }
  MonitorElement * me = store_->bookProfile(name,p1_temp) ;
  delete p1_temp ;
  return me ;
 }


