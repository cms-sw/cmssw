#include "Validation/EventGenerator/interface/DQMHelper.h"


DQMHelper::DQMHelper(DQMStore::IBooker *i):ibooker(i){

}

DQMHelper::~DQMHelper(){

}

MonitorElement* DQMHelper::book1dHisto(TString name,TString title,int n,double xmin,double xmax, 
				       TString xaxis, TString yaxis){
  MonitorElement* dqm=ibooker->book1D(std::string(name.Data()),std::string(title.Data()),n,xmin,xmax);
  dqm->getTH1()->Sumw2();
  dqm->setAxisTitle(std::string(xaxis.Data()),1);
  dqm->setAxisTitle(std::string(yaxis.Data()),2);
  return dqm;
}

MonitorElement* DQMHelper::book1dHisto(const std::string &name,const std::string &title,int n,double xmin,double xmax){
  MonitorElement* dqm=ibooker->book1D(name,title,n,xmin,xmax);
  dqm->getTH1()->Sumw2();
  return dqm;
}

MonitorElement* DQMHelper::book2dHisto(TString name,TString title,int nx,double xmin,double xmax,
				       int ny,double ymin,double ymax,TString xaxis, TString yaxis){
  MonitorElement* dqm=ibooker->book2D(std::string(name.Data()),std::string(title.Data()),nx,xmin,xmax,ny,ymin,ymax);
  dqm->getTH1()->Sumw2();
  dqm->setAxisTitle(std::string(xaxis.Data()),1);
  dqm->setAxisTitle(std::string(yaxis.Data()),2);
  return dqm;
}


MonitorElement* DQMHelper::book2dHisto(const std::string &name,const std::string &title,int nx,double xmin,double xmax,int ny,double ymin,double ymax){
  MonitorElement* dqm=ibooker->book2D(name,title,nx,xmin,xmax,ny,ymin,ymax);
  dqm->getTH1()->Sumw2();
  return dqm;
}



