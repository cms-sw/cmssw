#ifndef TopObjetcResolutionsJet_h
#define TopObjetcResolutionsJet_h

namespace res{
  class HelperJet {

  public:
    enum Flavor {kUds, kB};

    HelperJet(){};
    ~HelperJet(){};

    inline double pt (double pt, double eta,  Flavor flav);
    inline double eta(double pt, double eta,  Flavor flav);
    inline double phi(double pt, double eta,  Flavor flav);
    inline double a(double pt, double eta,  Flavor flav);
    inline double b(double pt, double eta,  Flavor flav);
    inline double c(double pt, double eta,  Flavor flav);
    inline double d(double pt, double eta,  Flavor flav);
    inline double et(double pt, double eta,  Flavor flav);
    inline double theta(double pt, double eta,  Flavor flav);
  };
}

inline double res::HelperJet::pt(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(fabs(eta)<1.4) res+= 6.68;
  else res+=-3.14*fabs(eta)+11.89;
  if(flav==kB){
    res=0.33*sqrt(pt);
    if(fabs(eta)<1.4) res+= 6.57;
    else res+=-1.09*fabs(eta)+8.50;    
  }
  return res;
}

inline double res::HelperJet::a(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 0.142188+0.288157*exp(-(0.0275628*pt));
		else if(fabs(eta)<0.35) res = 0.141215+0.277502*exp(-(0.0282327*pt));
		else if(fabs(eta)<0.5) res = 0.140493+0.26136*exp(-(0.0254784*pt));
		else if(fabs(eta)<0.7) res = 0.142608+0.371256*exp(-(0.0299132*pt));
		else if(fabs(eta)<0.9) res = 0.138817+0.237167*exp(-(0.0268413*pt));
		else if(fabs(eta)<1.15) res = 0.137322+0.293834*exp(-(0.0264771*pt));
		else if(fabs(eta)<1.4) res = 0.147184+0.40834*exp(-(0.0366769*pt));
		else if(fabs(eta)<1.7) res = 0.141529+0.195563*exp(-(0.023933*pt));
		else if(fabs(eta)<2.1) res = 0.129132+0.261012*exp(-(0.0323658*pt));
		else if(fabs(eta)<2.5) res = 0.120065+0.239351*exp(-(0.0356176*pt));
	}else{
		if(fabs(eta)<0.17) res = 0.142145+0.274541*exp(-(0.0286487*pt));
		else if(fabs(eta)<0.35) res = 0.146112+0.284777*exp(-(0.031039*pt));
		else if(fabs(eta)<0.5) res = 0.146703+0.343982*exp(-(0.0333917*pt));
		else if(fabs(eta)<0.7) res = 0.149067+0.241197*exp(-(0.0303415*pt));
		else if(fabs(eta)<0.9) res = 0.141233+0.304052*exp(-(0.0314644*pt));
		else if(fabs(eta)<1.15) res = 0.144531+0.266871*exp(-(0.0283385*pt));
		else if(fabs(eta)<1.4) res = 0.156593+0.41251*exp(-(0.0411896*pt));
		else if(fabs(eta)<1.7) res = 0.13452+0.269*exp(-(0.0230171*pt));
		else if(fabs(eta)<2.1) res = 0.0969416+0.211636*exp(-(0.0166117*pt));
		else if(fabs(eta)<2.5) res = 0.0727398+0.205132*exp(-(0.01661*pt));
	}
	return res;
}

inline double res::HelperJet::b(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 2.02521+0.174301*exp(-(-0.0136646*pt));
		else if(fabs(eta)<0.35) res = 0.690872+1.32174*exp(-(-0.00459989*pt));
		else if(fabs(eta)<0.5) res = -956.522+958.486*exp(-(-7.57551e-06*pt));
		else if(fabs(eta)<0.7) res = 2.01017+0.0670861*exp(-(-0.0194879*pt));
		else if(fabs(eta)<0.9) res = 1.92701+0.124384*exp(-(-0.0144246*pt));
		else if(fabs(eta)<1.15) res = 1.94854+0.034138*exp(-(-0.0227995*pt));
		else if(fabs(eta)<1.4) res = 2.07599+0.0208406*exp(-(-0.0214439*pt));
		else if(fabs(eta)<1.7) res = 1.99002+0.0077067*exp(-(-0.0278379*pt));
		else if(fabs(eta)<2.1) res = 1.93181+0.00196985*exp(-(-0.0390155*pt));
		else if(fabs(eta)<2.5) res = 1.75611+0.0287099*exp(-(-0.0202162*pt));
	}else{
		if(fabs(eta)<0.17) res = -644.18+645.342*exp(-(-1.7462e-05*pt));
		else if(fabs(eta)<0.35) res = -1.34986+2.48937*exp(-(-0.00396448*pt));
		else if(fabs(eta)<0.5) res = -4.8078+5.98195*exp(-(-0.0015*pt));
		else if(fabs(eta)<0.7) res = -9.55949+10.7419*exp(-(-0.000891802*pt));
		else if(fabs(eta)<0.9) res = 0.374264+0.884391*exp(-(-0.00642501*pt));
		else if(fabs(eta)<1.15) res = -9.51561+10.7429*exp(-(-0.000639173*pt));
		else if(fabs(eta)<1.4) res = -0.158886+1.43841*exp(-(-0.00347282*pt));
		else if(fabs(eta)<1.7) res = 0.28876+0.992659*exp(-(-0.00454673*pt));
		else if(fabs(eta)<2.1) res = 0.292201+0.922609*exp(-(-0.00536289*pt));
		else if(fabs(eta)<2.5) res = -657.479+658.533*exp(-(-1.40481e-05*pt));
	}
	return res;
}

inline double res::HelperJet::c(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = -2048.11+2051.27*exp(-(1.81337e-06*pt));
		else if(fabs(eta)<0.35) res = 2.86314+0.598484*exp(-(0.0322582*pt));
		else if(fabs(eta)<0.5) res = 2.54785+0.77807*exp(-(0.0100979*pt));
		else if(fabs(eta)<0.7) res = 2.78865+6.82558*exp(-(0.0677953*pt));
		else if(fabs(eta)<0.9) res = 2.63261+2.66562*exp(-(0.044099*pt));
		else if(fabs(eta)<1.15) res = 2.51691+2.75456*exp(-(0.0425201*pt));
		else if(fabs(eta)<1.4) res = 2.60458+0.623776*exp(-(0.0148415*pt));
		else if(fabs(eta)<1.7) res = 2.39005+1.60119*exp(-(0.0154536*pt));
		else if(fabs(eta)<2.1) res = 2.49783+0.408896*exp(-(0.04005*pt));
		else if(fabs(eta)<2.5) res = 2.02143+0.000354016*exp(-(-0.0476218*pt));
	}else{
		if(fabs(eta)<0.17) res = 1.72736+0.212526*exp(-(-0.00672611*pt));
		else if(fabs(eta)<0.35) res = -3480.32+3482.17*exp(-(-7.08939e-07*pt));
		else if(fabs(eta)<0.5) res = 1.89173+0.0381423*exp(-(-0.0151248*pt));
		else if(fabs(eta)<0.7) res = 2+0.00575323*exp(-(-0.0233459*pt));
		else if(fabs(eta)<0.9) res = 1.92618+0.00541047*exp(-(-0.0269683*pt));
		else if(fabs(eta)<1.15) res = 1.95535+0.0194328*exp(-(-0.0175058*pt));
		else if(fabs(eta)<1.4) res = 1.99855+0.00403398*exp(-(-0.027822*pt));
		else if(fabs(eta)<1.7) res = 2.02356+1.33745e-15*exp(-(-0.201028*pt));
		else if(fabs(eta)<2.1) res = 1.70305+0.0428122*exp(-(-0.0206051*pt));
		else if(fabs(eta)<2.5) res = 0.964128+0.370407*exp(-(-0.0119056*pt));
	}
	return res;
}
	
inline double res::HelperJet::d(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 0.144545+0.374307*exp(-(0.0314642*pt));
		else if(fabs(eta)<0.35) res = 0.141366+0.241457*exp(-(0.0263273*pt));
		else if(fabs(eta)<0.5) res = 0.142509+0.324478*exp(-(0.0288208*pt));
		else if(fabs(eta)<0.7) res = 0.139809+0.292392*exp(-(0.0257797*pt));
		else if(fabs(eta)<0.9) res = 0.137762+0.258213*exp(-(0.0265974*pt));
		else if(fabs(eta)<1.15) res = 0.135831+0.338346*exp(-(0.0277574*pt));
		else if(fabs(eta)<1.4) res = 0.148253+0.53999*exp(-(0.0415757*pt));
		else if(fabs(eta)<1.7) res = 0.145785+0.297049*exp(-(0.0319928*pt));
		else if(fabs(eta)<2.1) res = 0.126895+0.291297*exp(-(0.0328976*pt));
		else if(fabs(eta)<2.5) res = 0.119259+0.228403*exp(-(0.0341018*pt));
	}else{
		if(fabs(eta)<0.17) res = 0.139738+0.310387*exp(-(0.0296992*pt));
		else if(fabs(eta)<0.35) res = 0.142229+0.274556*exp(-(0.0286575*pt));
		else if(fabs(eta)<0.5) res = 0.14344+0.313425*exp(-(0.0294567*pt));
		else if(fabs(eta)<0.7) res = 0.147284+0.252005*exp(-(0.0303533*pt));
		else if(fabs(eta)<0.9) res = 0.13541+0.261376*exp(-(0.0260026*pt));
		else if(fabs(eta)<1.15) res = 0.144023+0.27422*exp(-(0.0282282*pt));
		else if(fabs(eta)<1.4) res = 0.158694+0.352571*exp(-(0.0385597*pt));
		else if(fabs(eta)<1.7) res = 0.135936+0.24773*exp(-(0.0226282*pt));
		else if(fabs(eta)<2.1) res = 0.0957325+0.213108*exp(-(0.016424*pt));
		else if(fabs(eta)<2.5) res = 0.0622281+0.185045*exp(-(0.0129321*pt));
	}
	return res;
}
	
inline double res::HelperJet::theta(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 0.0348657+0.0966067*exp(-(0.0388496*pt));
		else if(fabs(eta)<0.35) res = 0.0327521+0.0901303*exp(-(0.0354149*pt));
		else if(fabs(eta)<0.5) res = 0.030283+0.0970732*exp(-(0.036806*pt));
		else if(fabs(eta)<0.7) res = 0.0266898+0.0994089*exp(-(0.0379773*pt));
		else if(fabs(eta)<0.9) res = 0.022614+0.091974*exp(-(0.0384986*pt));
		else if(fabs(eta)<1.15) res = 0.0173219+0.0923928*exp(-(0.0395886*pt));
		else if(fabs(eta)<1.4) res = 0.0129851+0.077473*exp(-(0.0363236*pt));
		else if(fabs(eta)<1.7) res = 0.00967747+0.0667463*exp(-(0.0383101*pt));
		else if(fabs(eta)<2.1) res = 0.00744593+0.050027*exp(-(0.0432323*pt));
		else if(fabs(eta)<2.5) res = 0.00589309+0.0346904*exp(-(0.0484705*pt));
	}else{
		if(fabs(eta)<0.17) res = 0.0384999+0.105382*exp(-(0.0440247*pt));
		else if(fabs(eta)<0.35) res = 0.0384758+0.086093*exp(-(0.0436464*pt));
		else if(fabs(eta)<0.5) res = 0.0325808+0.0860212*exp(-(0.0371074*pt));
		else if(fabs(eta)<0.7) res = 0.0304022+0.094116*exp(-(0.0402575*pt));
		else if(fabs(eta)<0.9) res = 0.0246649+0.0883167*exp(-(0.0386519*pt));
		else if(fabs(eta)<1.15) res = 0.0197668+0.080217*exp(-(0.0368397*pt));
		else if(fabs(eta)<1.4) res = 0.0148531+0.0692569*exp(-(0.0354383*pt));
		else if(fabs(eta)<1.7) res = 0.0113007+0.0601861*exp(-(0.0373138*pt));
		else if(fabs(eta)<2.1) res = 0.00860103+0.0522148*exp(-(0.0450405*pt));
		else if(fabs(eta)<2.5) res = 0.00692992+0.031671*exp(-(0.0490789*pt));
	}
	return res;
}
	
inline double res::HelperJet::phi(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 0.0217379+0.157228*exp(-(0.0237049*pt));
		else if(fabs(eta)<0.35) res = 0.0246735+0.165694*exp(-(0.0269728*pt));
		else if(fabs(eta)<0.5) res = 0.0226496+0.162875*exp(-(0.0250225*pt));
		else if(fabs(eta)<0.7) res = 0.0232004+0.167887*exp(-(0.0259427*pt));
		else if(fabs(eta)<0.9) res = 0.0220083+0.183889*exp(-(0.0273017*pt));
		else if(fabs(eta)<1.15) res = 0.0216601+0.18482*exp(-(0.027148*pt));
		else if(fabs(eta)<1.4) res = 0.0225191+0.19558*exp(-(0.0271747*pt));
		else if(fabs(eta)<1.7) res = 0.0208365+0.19241*exp(-(0.0262627*pt));
		else if(fabs(eta)<2.1) res = 0.0233749+0.180684*exp(-(0.032052*pt));
		else if(fabs(eta)<2.5) res = 0.0231891+0.165484*exp(-(0.0389916*pt));
	}else{
		if(fabs(eta)<0.17) res = 0.025945+0.16529*exp(-(0.027385*pt));
		else if(fabs(eta)<0.35) res = 0.0257294+0.155366*exp(-(0.0268639*pt));
		else if(fabs(eta)<0.5) res = 0.0231924+0.166508*exp(-(0.0264153*pt));
		else if(fabs(eta)<0.7) res = 0.0252736+0.176977*exp(-(0.0281594*pt));
		else if(fabs(eta)<0.9) res = 0.0269299+0.174505*exp(-(0.0295757*pt));
		else if(fabs(eta)<1.15) res = 0.0250191+0.178321*exp(-(0.0284006*pt));
		else if(fabs(eta)<1.4) res = 0.0242345+0.174682*exp(-(0.027202*pt));
		else if(fabs(eta)<1.7) res = 0.0242234+0.199735*exp(-(0.0291262*pt));
		else if(fabs(eta)<2.1) res = 0.0300017+0.175617*exp(-(0.0357093*pt));
		else if(fabs(eta)<2.5) res = 0.0282978+0.141994*exp(-(0.0383663*pt));
	}
	return res;
}
	
inline double res::HelperJet::et(double pt, double eta,  Flavor flav)
{
  double res = 0.29*sqrt(pt);
  if(flav==kB){ 
		if(fabs(eta)<0.17) res = 10.1445+0.0445196*pt;
		else if(fabs(eta)<0.35) res = 8.944+0.0576391*pt;
		else if(fabs(eta)<0.5) res = 8.8462+0.0630703*pt;
		else if(fabs(eta)<0.7) res = 8.90812+0.057605*pt;
		else if(fabs(eta)<0.9) res = 8.42307+0.0623024*pt;
		else if(fabs(eta)<1.15) res = 8.64735+0.0619689*pt;
		else if(fabs(eta)<1.4) res = 8.90574+0.0595526*pt;
		else if(fabs(eta)<1.7) res = 8.74415+0.0612412*pt;
		else if(fabs(eta)<2.1) res = 7.80525+0.0511843*pt;
		else if(fabs(eta)<2.5) res = 6.07877+0.0569574*pt;
	}else{
		if(fabs(eta)<0.17) res = 8.04345+0.0705188*pt;
		else if(fabs(eta)<0.35) res = 7.77565+0.0722247*pt;
		else if(fabs(eta)<0.5) res = 8.53558+0.0647725*pt;
		else if(fabs(eta)<0.7) res = 7.99374+0.0681428*pt;
		else if(fabs(eta)<0.9) res = 7.51695+0.0755128*pt;
		else if(fabs(eta)<1.15) res = 7.70804+0.0748981*pt;
		else if(fabs(eta)<1.4) res = 7.2996+0.0852397*pt;
		else if(fabs(eta)<1.7) res = 7.95923+0.0715722*pt;
		else if(fabs(eta)<2.1) res = 7.68012+0.0543317*pt;
		else if(fabs(eta)<2.5) res = 6.59141+0.0452384*pt;
	}
	return res;
}
	
inline double res::HelperJet::eta(double pt, double eta,  Flavor flav)
{
  double res=-1.53e-4*pt+0.05;
  if(flav==kB){
		if(fabs(eta)<0.17) res = 0.0354411+0.130211*exp(-(0.0423321*pt));
		else if(fabs(eta)<0.35) res = 0.0341742+0.115784*exp(-(0.0375787*pt));
		else if(fabs(eta)<0.5) res = 0.0336537+0.143109*exp(-(0.0416702*pt));
		else if(fabs(eta)<0.7) res = 0.031894+0.140572*exp(-(0.0405564*pt));
		else if(fabs(eta)<0.9) res = 0.0301507+0.126005*exp(-(0.0384231*pt));
		else if(fabs(eta)<1.15) res = 0.0268018+0.140956*exp(-(0.03872*pt));
		else if(fabs(eta)<1.4) res = 0.0242941+0.131887*exp(-(0.0337869*pt));
		else if(fabs(eta)<1.7) res = 0.0223985+0.142949*exp(-(0.0351412*pt));
		else if(fabs(eta)<2.1) res = 0.0229694+0.145993*exp(-(0.0391759*pt));
		else if(fabs(eta)<2.5) res = 0.0243171+0.145419*exp(-(0.0430824*pt));
	}else{
		if(fabs(eta)<0.17) res = 0.038575+0.129026*exp(-(0.0427157*pt));
		else if(fabs(eta)<0.35) res = 0.0400221+0.127222*exp(-(0.0458212*pt));
		else if(fabs(eta)<0.5) res = 0.0358472+0.119962*exp(-(0.0395586*pt));
		else if(fabs(eta)<0.7) res = 0.0359612+0.136094*exp(-(0.04212*pt));
		else if(fabs(eta)<0.9) res = 0.0327581+0.127339*exp(-(0.0393077*pt));
		else if(fabs(eta)<1.15) res = 0.0305095+0.116238*exp(-(0.0347537*pt));
		else if(fabs(eta)<1.4) res = 0.027529+0.117893*exp(-(0.032858*pt));
		else if(fabs(eta)<1.7) res = 0.0253065+0.112751*exp(-(0.0307028*pt));
		else if(fabs(eta)<2.1) res = 0.0266957+0.131998*exp(-(0.0380667*pt));
		else if(fabs(eta)<2.5) res = 0.0298413+0.13651*exp(-(0.0458475*pt));
	}
  return res;
}

#endif
