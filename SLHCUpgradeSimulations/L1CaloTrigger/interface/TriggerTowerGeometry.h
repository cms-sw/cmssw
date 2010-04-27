#ifndef TRIGGER_TOWER_GEO
#define TRIGGER_TOWER_GEO

class  TriggerTowerGeometry
{
   public:
  TriggerTowerGeometry()
    {
      
      for(int i=1;i<=20;++i)
	mappingEta_[i] =0.087;

      mappingEta_[21] = 0.09;
      mappingEta_[22] = 0.1;
      mappingEta_[23] = 0.113;
      mappingEta_[24] = 0.129;
      mappingEta_[25] = 0.15;
      mappingEta_[26] = 0.178;
      mappingEta_[27] = 0.15;
      mappingEta_[28] = 0.35;
    }

  double eta(int iEta)
    {
	  double eta=0;
	  for(int i=1;i<=abs(iEta);++i)
	    {
	      eta+=mappingEta_[i];
	    }
	  eta-=mappingEta_[abs(iEta)]/2;

	  if(iEta>0) return eta;
	  else
	    return -eta;
    }

  double phi(int iPhi)
    {
      return iPhi*0.087-0.087/2;
    }

  double towerEtaSize(int iEta)
    {
      return mappingEta_[abs(iEta)];
    }

  double towerPhiSize(int iPhi)
    {
      return 0.087;
    }

 private:
  std::map<int,double> mappingEta_;
};

#endif
