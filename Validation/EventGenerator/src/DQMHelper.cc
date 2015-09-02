#include "Validation/EventGenerator/interface/DQMHelper.h"


DQMHelper::DQMHelper(DQMStore::IBooker *i):ibooker(i){

}

DQMHelper::~DQMHelper(){

}

MonitorElement* DQMHelper::book1dHisto(std::string name,std::string title,int n,double xmin,double xmax, 
				       std::string xaxis, std::string yaxis){
  MonitorElement* dqm=ibooker->book1D(name,title,n,xmin,xmax);
  dqm->getTH1()->Sumw2();
  dqm->setAxisTitle(xaxis,1);
  dqm->setAxisTitle(yaxis,2);
  return dqm;
}

MonitorElement* DQMHelper::book1dHisto(const std::string &name,const std::string &title,int n,double xmin,double xmax){
  MonitorElement* dqm=ibooker->book1D(name,title,n,xmin,xmax);
  dqm->getTH1()->Sumw2();
  return dqm;
}

MonitorElement* DQMHelper::book2dHisto(std::string name,std::string title,int nx,double xmin,double xmax,
				       int ny,double ymin,double ymax,std::string xaxis, std::string yaxis){
  MonitorElement* dqm=ibooker->book2D(name,title,nx,xmin,xmax,ny,ymin,ymax);
  dqm->getTH1()->Sumw2();
  dqm->setAxisTitle(xaxis,1);
  dqm->setAxisTitle(yaxis,2);
  return dqm;
}


MonitorElement* DQMHelper::book2dHisto(const std::string &name,const std::string &title,int nx,double xmin,double xmax,int ny,double ymin,double ymax){
  MonitorElement* dqm=ibooker->book2D(name,title,nx,xmin,xmax,ny,ymin,ymax);
  dqm->getTH1()->Sumw2();
  return dqm;
}



