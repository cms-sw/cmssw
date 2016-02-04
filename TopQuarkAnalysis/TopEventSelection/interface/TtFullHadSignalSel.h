#ifndef TtFullHadSignalSel_h
#define TtFullHadSignalSel_h

#include <vector>
#include "TMath.h"
#include "Math/VectorUtil.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"

class TtFullHadSignalSel {
  // common calculator class for likelihood
  // variables in fully hadronic ttbar decays
public:

  TtFullHadSignalSel();
  TtFullHadSignalSel(const std::vector<pat::Jet>&);
  ~TtFullHadSignalSel();
  
  double H()      const { return H_;      }
  double Ht()     const { return Ht_;     }
  double Ht123()  const { return Ht123_;  }
  double Ht3jet() const { return Ht3jet_; }
  double sqrt_s() const { return sqrt_s_; }
  double Et56()   const { return Et56_;   }
  double M3()     const { return M3_;     }

  double TCHP_Bjets() const { return TCHP_Bjets_; }
  double SSV_Bjets()  const { return SSV_Bjets_;  }
  double CSV_Bjets()  const { return CSV_Bjets_;  }
  double SM_Bjets()   const { return SM_Bjets_;   }

  double TCHP_Bjet1() const { return TCHP_Bjet1_; }
  double TCHP_Bjet2() const { return TCHP_Bjet2_; }
  double TCHP_Bjet3() const { return TCHP_Bjet3_; }
  double TCHP_Bjet4() const { return TCHP_Bjet4_; }
  double TCHP_Bjet5() const { return TCHP_Bjet5_; }
  double TCHP_Bjet6() const { return TCHP_Bjet6_; }
  double SSV_Bjet1()  const { return SSV_Bjet1_;  }
  double SSV_Bjet2()  const { return SSV_Bjet2_;  }
  double SSV_Bjet3()  const { return SSV_Bjet3_;  }
  double SSV_Bjet4()  const { return SSV_Bjet4_;  }
  double SSV_Bjet5()  const { return SSV_Bjet5_;  }
  double SSV_Bjet6()  const { return SSV_Bjet6_;  }
  double CSV_Bjet1()  const { return CSV_Bjet1_;  }
  double CSV_Bjet2()  const { return CSV_Bjet2_;  }
  double CSV_Bjet3()  const { return CSV_Bjet3_;  }
  double CSV_Bjet4()  const { return CSV_Bjet4_;  }
  double CSV_Bjet5()  const { return CSV_Bjet5_;  }
  double CSV_Bjet6()  const { return CSV_Bjet6_;  }
  double SM_Bjet1()   const { return SM_Bjet1_;   }
  double SM_Bjet2()   const { return SM_Bjet2_;   }
  double SM_Bjet3()   const { return SM_Bjet3_;   }
  double SM_Bjet4()   const { return SM_Bjet4_;   }
  double SM_Bjet5()   const { return SM_Bjet5_;   }
  double SM_Bjet6()   const { return SM_Bjet6_;   }

  double pt1() const { return pt1_; }
  double pt2() const { return pt2_; }
  double pt3() const { return pt3_; }
  double pt4() const { return pt4_; }
  double pt5() const { return pt5_; }
  double pt6() const { return pt6_; }

  double pt1_pt2() const { return (pt1_ - pt2_); }
  double pt1_pt3() const { return (pt1_ - pt3_); }
  double pt1_pt4() const { return (pt1_ - pt4_); }
  double pt1_pt5() const { return (pt1_ - pt5_); }
  double pt1_pt6() const { return (pt1_ - pt6_); }
  double pt2_pt3() const { return (pt2_ - pt3_); }
  double pt2_pt4() const { return (pt2_ - pt4_); }
  double pt2_pt5() const { return (pt2_ - pt5_); }
  double pt2_pt6() const { return (pt2_ - pt6_); }
  double pt3_pt4() const { return (pt3_ - pt4_); }
  double pt3_pt5() const { return (pt3_ - pt5_); }
  double pt3_pt6() const { return (pt3_ - pt6_); }
  double pt4_pt5() const { return (pt4_ - pt5_); }
  double pt4_pt6() const { return (pt4_ - pt6_); }
  double pt5_pt6() const { return (pt5_ - pt6_); }

  double pt1_pt2_norm() const { return ((pt1_ - pt2_)/(pt1_ + pt2_)); }
  double pt1_pt3_norm() const { return ((pt1_ - pt3_)/(pt1_ + pt3_)); }
  double pt1_pt4_norm() const { return ((pt1_ - pt4_)/(pt1_ + pt4_)); }
  double pt1_pt5_norm() const { return ((pt1_ - pt5_)/(pt1_ + pt5_)); }
  double pt1_pt6_norm() const { return ((pt1_ - pt6_)/(pt1_ + pt6_)); }
  double pt2_pt3_norm() const { return ((pt2_ - pt3_)/(pt2_ + pt3_)); }
  double pt2_pt4_norm() const { return ((pt2_ - pt4_)/(pt2_ + pt4_)); }
  double pt2_pt5_norm() const { return ((pt2_ - pt5_)/(pt2_ + pt5_)); }
  double pt2_pt6_norm() const { return ((pt2_ - pt6_)/(pt2_ + pt6_)); }
  double pt3_pt4_norm() const { return ((pt3_ - pt4_)/(pt3_ + pt4_)); }
  double pt3_pt5_norm() const { return ((pt3_ - pt5_)/(pt3_ + pt5_)); }
  double pt3_pt6_norm() const { return ((pt3_ - pt6_)/(pt3_ + pt6_)); }
  double pt4_pt5_norm() const { return ((pt4_ - pt5_)/(pt4_ + pt5_)); }
  double pt4_pt6_norm() const { return ((pt4_ - pt6_)/(pt4_ + pt6_)); }
  double pt5_pt6_norm() const { return ((pt5_ - pt6_)/(pt5_ + pt6_)); }

  double jet1_etaetaMoment() const { return jet1_etaetaMoment_; }
  double jet2_etaetaMoment() const { return jet2_etaetaMoment_; }
  double jet3_etaetaMoment() const { return jet3_etaetaMoment_; }
  double jet4_etaetaMoment() const { return jet4_etaetaMoment_; }
  double jet5_etaetaMoment() const { return jet5_etaetaMoment_; }
  double jet6_etaetaMoment() const { return jet6_etaetaMoment_; }
  double jet1_etaphiMoment() const { return jet1_etaphiMoment_; }
  double jet2_etaphiMoment() const { return jet2_etaphiMoment_; }
  double jet3_etaphiMoment() const { return jet3_etaphiMoment_; }
  double jet4_etaphiMoment() const { return jet4_etaphiMoment_; }
  double jet5_etaphiMoment() const { return jet5_etaphiMoment_; }
  double jet6_etaphiMoment() const { return jet6_etaphiMoment_; }
  double jet1_phiphiMoment() const { return jet1_phiphiMoment_; }
  double jet2_phiphiMoment() const { return jet2_phiphiMoment_; }
  double jet3_phiphiMoment() const { return jet3_phiphiMoment_; }
  double jet4_phiphiMoment() const { return jet4_phiphiMoment_; }
  double jet5_phiphiMoment() const { return jet5_phiphiMoment_; }
  double jet6_phiphiMoment() const { return jet6_phiphiMoment_; }

  double jets_etaetaMoment() const { return (jet1_etaetaMoment_ + jet2_etaetaMoment_ + jet3_etaetaMoment_ + jet4_etaetaMoment_ + jet5_etaetaMoment_ + jet6_etaetaMoment_); }
  double jets_etaphiMoment() const { return (jet1_etaphiMoment_ + jet2_etaphiMoment_ + jet3_etaphiMoment_ + jet4_etaphiMoment_ + jet5_etaphiMoment_ + jet6_etaphiMoment_); }
  double jets_phiphiMoment() const { return (jet1_phiphiMoment_ + jet2_phiphiMoment_ + jet3_phiphiMoment_ + jet4_phiphiMoment_ + jet5_phiphiMoment_ + jet6_phiphiMoment_); }

  double aplanarity()  const { return aplanarity_;  }
  double sphericity()  const { return sphericity_;  }
  double circularity() const { return circularity_; }
  double isotropy()    const { return isotropy_;    }
  double C()           const { return C_;           }
  double D()           const { return D_;           }
  double centrality()  const { return (Ht_/H_);     }

  double dRMin1()        const { return dRMin1_;        }
  double dRMin2()        const { return dRMin2_;        }
  double sumDR3JetMin1() const { return sumDR3JetMin1_; }
  double sumDR3JetMin2() const { return sumDR3JetMin2_; }
	 			   	    			   
  double dRMin1Mass()        const { return dRMin1Mass_;        }
  double dRMin2Mass()        const { return dRMin2Mass_;        }
  double sumDR3JetMin1Mass() const { return sumDR3JetMin1Mass_; }
  double sumDR3JetMin2Mass() const { return sumDR3JetMin2Mass_; }

private:
  
  double H_;
  double Ht_;
  double Ht123_;
  double Ht3jet_;
  double sqrt_s_;
  double Et56_;
  double M3_;
  
  double TCHP_Bjets_;
  double SSV_Bjets_;
  double CSV_Bjets_;
  double SM_Bjets_;

  double TCHP_Bjet1_;
  double TCHP_Bjet2_;
  double TCHP_Bjet3_;
  double TCHP_Bjet4_;
  double TCHP_Bjet5_;
  double TCHP_Bjet6_;
  double SSV_Bjet1_;
  double SSV_Bjet2_;
  double SSV_Bjet3_;
  double SSV_Bjet4_;
  double SSV_Bjet5_;
  double SSV_Bjet6_;
  double CSV_Bjet1_;
  double CSV_Bjet2_;
  double CSV_Bjet3_;
  double CSV_Bjet4_;
  double CSV_Bjet5_;
  double CSV_Bjet6_;
  double SM_Bjet1_;
  double SM_Bjet2_;
  double SM_Bjet3_;
  double SM_Bjet4_;
  double SM_Bjet5_;
  double SM_Bjet6_;

  double pt1_;
  double pt2_;
  double pt3_;
  double pt4_;
  double pt5_;
  double pt6_;

  double jet1_etaetaMoment_;
  double jet2_etaetaMoment_;
  double jet3_etaetaMoment_;
  double jet4_etaetaMoment_;
  double jet5_etaetaMoment_;
  double jet6_etaetaMoment_;
  double jet1_etaphiMoment_;
  double jet2_etaphiMoment_;
  double jet3_etaphiMoment_;
  double jet4_etaphiMoment_;
  double jet5_etaphiMoment_;
  double jet6_etaphiMoment_;
  double jet1_phiphiMoment_;
  double jet2_phiphiMoment_;
  double jet3_phiphiMoment_;
  double jet4_phiphiMoment_;
  double jet5_phiphiMoment_;
  double jet6_phiphiMoment_;

  double aplanarity_;
  double sphericity_;
  double circularity_;
  double isotropy_;
  double C_;
  double D_;

  double dRMin1_;
  double dRMin2_;
  double sumDR3JetMin1_;
  double sumDR3JetMin2_;

  double dRMin1Mass_;
  double dRMin2Mass_;
  double sumDR3JetMin1Mass_;
  double sumDR3JetMin2Mass_;
};

#endif
