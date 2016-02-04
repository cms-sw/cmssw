#ifndef TopObjetcResolutionsElectron_h
#define TopObjetcResolutionsElectron_h

namespace res{
  class HelperElectron {

  public:
    HelperElectron(){};
    ~HelperElectron(){};

    inline double pt (double pt, double eta);
    inline double eta(double pt, double eta);
    inline double phi(double pt, double eta);
	  inline double a(double pt, double eta);
    inline double b(double pt, double eta);
    inline double c(double pt, double eta);
    inline double d(double pt, double eta);
    inline double et(double pt, double eta);
    inline double theta(double pt, double eta);
  };
}

inline double res::HelperElectron::pt(double pt, double eta)
{
  double res=0.2*sqrt(pt);
  if( fabs(eta)<=0.8 ) 
    res+=-0.28*fabs(eta)+0.54;
  else if( 0.8<fabs(eta) && fabs(eta)<=1.4 ) 
    res+= 1.52*fabs(eta)-1.07;
  else  
    res+=-0.158*eta*eta +0.97;
  return res;
}

inline double res::HelperElectron::a(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = 0.0114228+0.1135*exp(-(0.111787*pt));
	else if(fabs(eta)<0.35) res = 0.010297+0.0163219*exp(-(0.0248655*pt));
	else if(fabs(eta)<0.5) res = 0.0081634+0.0230718*exp(-(0.0250963*pt));
	else if(fabs(eta)<0.7) res = 0.0122627+0.0228103*exp(-(0.0420525*pt));
	else if(fabs(eta)<0.9) res = 0.0124118+0.0384051*exp(-(0.0545988*pt));
	else if(fabs(eta)<1.15) res = 0.0133398+0.0307574*exp(-(0.0316605*pt));
	else if(fabs(eta)<1.3) res = 0.0144956+0.0355239*exp(-(0.0273916*pt));
	else if(fabs(eta)<1.6) res = -13.9017+13.9315*exp(-(7.01823e-06*pt));
	else if(fabs(eta)<1.9) res = 0.0106309+0.0230149*exp(-(0.00792017*pt));
	else if(fabs(eta)<2.5) res = 0.0187822+0.119922*exp(-(0.140598*pt));
  return res;
}

inline double res::HelperElectron::b(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = -36.5147+36.516*exp(-(-9.36847e-06*pt));
	else if(fabs(eta)<0.35) res = -0.132294+0.134289*exp(-(-0.0021212*pt));
	else if(fabs(eta)<0.5) res = -96.9689+96.9704*exp(-(-2.83158e-06*pt));
	else if(fabs(eta)<0.7) res = -122.755+122.757*exp(-(-1.90859e-06*pt));
	else if(fabs(eta)<0.9) res = -226.455+226.457*exp(-(-8.91395e-07*pt));
	else if(fabs(eta)<1.15) res = -249.279+249.281*exp(-(-7.57645e-07*pt));
	else if(fabs(eta)<1.3) res = -16.7465+16.7481*exp(-(-1.17848e-05*pt));
	else if(fabs(eta)<1.6) res = -128.535+128.537*exp(-(-1.52236e-06*pt));
	else if(fabs(eta)<1.9) res = -66.1731+66.1762*exp(-(-2.49121e-06*pt));
	else if(fabs(eta)<2.5) res = -0.0199509+0.0237796*exp(-(-0.00591733*pt));
  return res;
}

inline double res::HelperElectron::c(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = -68.6101+68.6161*exp(-(-7.88064e-07*pt));
	else if(fabs(eta)<0.35) res = -130.219+130.224*exp(-(-8.39696e-07*pt));
	else if(fabs(eta)<0.5) res = -262.825+262.83*exp(-(-4.26008e-07*pt));
	else if(fabs(eta)<0.7) res = -47.898+47.9041*exp(-(-2.07568e-06*pt));
	else if(fabs(eta)<0.9) res = -178.795+178.799*exp(-(-5.78263e-07*pt));
	else if(fabs(eta)<1.15) res = -249.26+249.267*exp(-(-3.16408e-07*pt));
	else if(fabs(eta)<1.3) res = -173.603+173.606*exp(-(-1.58982e-06*pt));
	else if(fabs(eta)<1.6) res = -98.726+98.7326*exp(-(-2.67151e-06*pt));
	else if(fabs(eta)<1.9) res = -82.6028+82.6122*exp(-(-2.79483e-06*pt));
	else if(fabs(eta)<2.5) res = -119.94+119.95*exp(-(-1.69882e-06*pt));
  return res;
}

inline double res::HelperElectron::d(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = 0.0114197+0.081753*exp(-(0.0969625*pt));
	else if(fabs(eta)<0.35) res = 0.0097562+0.0137737*exp(-(0.0187112*pt));
	else if(fabs(eta)<0.5) res = 0.00844899+0.021595*exp(-(0.0241679*pt));
	else if(fabs(eta)<0.7) res = 0.0124758+0.0243678*exp(-(0.0459914*pt));
	else if(fabs(eta)<0.9) res = 0.0117518+0.0357218*exp(-(0.0463595*pt));
	else if(fabs(eta)<1.15) res = 0.0133048+0.0292511*exp(-(0.0305812*pt));
	else if(fabs(eta)<1.3) res = 0.0144949+0.0337369*exp(-(0.0269872*pt));
	else if(fabs(eta)<1.6) res = 0.0137634+0.0343427*exp(-(0.0208025*pt));
	else if(fabs(eta)<1.9) res = 0.00572644+0.0270719*exp(-(0.00536847*pt));
	else if(fabs(eta)<2.5) res = 0.0189177+0.169591*exp(-(0.152597*pt));
  return res;
}

inline double res::HelperElectron::theta(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = 0.000282805+0.000157786*exp(-(0.0343273*pt));
	else if(fabs(eta)<0.35) res = 0.000184362+4.34076e-05*exp(-(-0.0131909*pt));
	else if(fabs(eta)<0.5) res = 0.000249332+5.83114e-05*exp(-(0.0508729*pt));
	else if(fabs(eta)<0.7) res = -6.56357e-05+0.000325051*exp(-(0.00177319*pt));
	else if(fabs(eta)<0.9) res = 0.000182277+0.000125324*exp(-(0.0581923*pt));
	else if(fabs(eta)<1.15) res = 0.000140771+0.000407914*exp(-(0.0971668*pt));
	else if(fabs(eta)<1.3) res = 0.000125551+0.001266*exp(-(0.180176*pt));
	else if(fabs(eta)<1.6) res = 0.000107631+101920*exp(-(1.17024*pt));
	else if(fabs(eta)<1.9) res = 8.33927e-05+158936*exp(-(1.20127*pt));
	else if(fabs(eta)<2.5) res = 6.55271e-05+0.12459*exp(-(0.437044*pt));
  return res;
}

inline double res::HelperElectron::phi(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = 0.000175676+0.000471783*exp(-(0.0383161*pt));
	else if(fabs(eta)<0.35) res = 0.000202185+0.00048635*exp(-(0.0373331*pt));
	else if(fabs(eta)<0.5) res = 0.000150868+0.000444216*exp(-(0.0268835*pt));
	else if(fabs(eta)<0.7) res = 0.000243624+0.00182347*exp(-(0.0850746*pt));
	else if(fabs(eta)<0.9) res = 0.000254463+0.000431233*exp(-(0.0446507*pt));
	else if(fabs(eta)<1.15) res = 0.000309592+0.000918965*exp(-(0.0555677*pt));
	else if(fabs(eta)<1.3) res = 0.000502204+0.000277996*exp(-(0.076721*pt));
	else if(fabs(eta)<1.6) res = 0.000361181+0.000655126*exp(-(0.0238519*pt));
	else if(fabs(eta)<1.9) res = 0.000321587+0.00155721*exp(-(0.0337709*pt));
	else if(fabs(eta)<2.5) res = 0.000819101+0.00205336*exp(-(0.0992806*pt));
  return res;
}

inline double res::HelperElectron::et(double pt, double eta)
{
  double res = 1000.; 
	if(fabs(eta)<0.17) res = 0.326238+0.00760789*pt;
	else if(fabs(eta)<0.35) res = 0.40493+0.00659958*pt;
	else if(fabs(eta)<0.5) res = 0.369785+0.00690331*pt;
	else if(fabs(eta)<0.7) res = 0.437539+0.00703808*pt;
	else if(fabs(eta)<0.9) res = 0.456138+0.0078252*pt;
	else if(fabs(eta)<1.15) res = 0.518685+0.00907836*pt;
	else if(fabs(eta)<1.3) res = 0.733672+0.00953255*pt;
	else if(fabs(eta)<1.6) res = 1.02678+0.0116056*pt;
	else if(fabs(eta)<1.9) res = 0.948368+0.00977619*pt;
	else if(fabs(eta)<2.5) res = 0.418302+0.0127816*pt;
  return res;
}

inline double res::HelperElectron::eta(double pt, double eta)
{
  double res = 1000.;
	if(fabs(eta)<0.17) res = 0.000266154+0.000104322*exp(-(0.0140464*pt));
	else if(fabs(eta)<0.35) res = -0.251539+0.251791*exp(-(-7.37147e-07*pt));
	else if(fabs(eta)<0.5) res = 0.000290074+1.54664e-12*exp(-(-0.115541*pt));
	else if(fabs(eta)<0.7) res = 8.37182e-05+0.000233453*exp(-(0.00602386*pt));
	else if(fabs(eta)<0.9) res = 0.000229422+0.000114253*exp(-(0.0188935*pt));
	else if(fabs(eta)<1.15) res = 0.000191525+0.000404238*exp(-(0.0554545*pt));
	else if(fabs(eta)<1.3) res = 0.000195461+1.43699e-07*exp(-(-0.0315088*pt));
	else if(fabs(eta)<1.6) res = 0.000223422+2.05169e+07*exp(-(1.41408*pt));
	else if(fabs(eta)<1.9) res = -0.867114+0.867336*exp(-(4.08639e-07*pt));
	else if(fabs(eta)<2.5) res = -0.866567+0.866746*exp(-(-8.62871e-07*pt));
  return res;
}

#endif
