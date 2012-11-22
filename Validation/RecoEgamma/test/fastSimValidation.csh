#!/bin/csh

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  
#  
#  "Validation/RecoEgamma/test/PhotonValidator_cfg.py
#
# The default list of histograms (configurable) is based on version VXX-XX-XX
# of Validation/RecoEgamma
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================
setenv TYPE Photons
setenv CMSSWver1 6_1_0
setenv RELEASE 6_1_0
setenv PRERELEASE pre5

setenv FULLGLOBALTAG START61_V4-v1
setenv FASTGLOBALTAG START61_V4_FastSim-v1

setenv RELEASE ${RELEASE}_${PRERELEASE}
#setenv RELEASE ${RELEASE}


#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${PRERELEASE}/src/Validation/RecoEgamma/test


#Name of sample (affects output directory name and htmldescription only) 


setenv SAMPLE H130GGgluonfusion

#TYPE must be one ofPixelMatchGsfElectron, Photon 

#==============END BASIC CONFIGURATION==================


#Input root trees for the two cases to be compared 


if ($SAMPLE == SingleGammaPt10IDEAL) then

setenv FULLSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaPt10.root
setenv FASTSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaPt10_FastSim.root

else if ($SAMPLE == SingleGammaPt35IDEAL) then 

setenv FULLSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaPt35.root
setenv FASTSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaPt35_FastSim.root


else if ($SAMPLE == SingleGammaFlatPt10_100IDEAL) then 

setenv FULLSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaFlatPt10To100.root
setenv FASTSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_SingleGammaFlatPt10To100_FastSim.root


else if ($SAMPLE == H130GGgluonfusion) then 

setenv HISTOPATHNAME_Efficiencies DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/Efficiencies
setenv HISTOPATHNAME_Photons DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/Photons
setenv HISTOPATHNAME_Conversions DQMData/Run\ 1/EgammaV/Run\ summary/PhotonValidator/ConversionInfo
setenv FULLSIM ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${RELEASE}-${FULLGLOBALTAG}__DQM.root
setenv FASTSIM ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${RELEASE}-${FASTGLOBALTAG}__GEN-SIM-DIGI-RECO.root



else if ($SAMPLE == PhotonJets_Pt_10) then

setenv FULLSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_PhotonJets_Pt_10.root
setenv FASTSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_PhotonJets_Pt_10_FastSim.root


else if ($SAMPLE ==  GammaJets_Pt_80_120STARTUP) then 

setenv FULLSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_GammaJets_Pt_80_120.root
setenv FASTSIM ${WorkDir1}/PhotonValidationRelVal${RELEASE}_GammaJets_Pt_80_120.root


else if ($SAMPLE == QCD_Pt_80_120STARTUP) then 

endif



#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation
cd $OUTPATH
if (! -d $RELEASE) then
  mkdir $RELEASE
endif
setenv OUTPATH $OUTPATH/$RELEASE
cd $OUTPATH

if (! -d ${TYPE}) then
  mkdir ${TYPE}
endif
setenv OUTPATH $OUTPATH/${TYPE}
cd  $OUTPATH

if (! -d vsFastSim) then
  mkdir vsFastSim
endif
setenv OUTPATH $OUTPATH/vsFastSim

setenv OUTDIR $OUTPATH/${SAMPLE}
if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR


#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == Photons ) then


cat > efficiencyForPhotons <<EOF
  recoEffVsEta
  recoEffVsPhi
  recoEffVsEt
  deadChVsEta
  deadChVsPhi
  deadChVsEt
EOF


#  gamgamMassAll
#  gamgamMassBarrel
#  gamgamMassEndcap
#  gamgamMassNoConvAll
#  gamgamMassNoConvBarrel
#  gamgamMassNoConvEndcap
#  gamgamMassConvAll
#  gamgamMassConvBarrel
#  gamgamMassConvEndcap


cat > scaledhistosForPhotons <<EOF

  scEta
  scPhi
  scEAll
  scEtAll
  phoEta
  phoPhi
  phoDEta
  phoDPhi
  phoEAll
  phoEtAll
  eResAll
  eResBarrel
  eResEndcap
  eResunconvAll
  eResunconvBarrel
  eResunconvEndcap
  eResconvAll
  eResconvBarrel
  eResconvEndcap
  r9All
  r9Barrel
  r9Endcap
  r1All
  r1Barrel
  r1Endcap
  r2All
  r2Barrel
  r2Endcap
  sigmaIetaIetaAll
  sigmaIetaIetaBarrel
  sigmaIetaIetaEndcap
  ecalRecHitSumEtConeDR04Barrel
  ecalRecHitSumEtConeDR04Endcap
  isoTrkSolidConeDR04All
  isoTrkSolidConeDR04Barrel
  isoTrkSolidConeDR04Endcap
  nTrkSolidConeDR04All
  nTrkSolidConeDR04Barrel
  nTrkSolidConeDR04Endcap



EOF

cat > scaledhistosForPhotonsLogScale <<EOF
  hOverEAll
  hOverEBarrel
  hOverEEndcap
  newhOverEAll
  newhOverEBarrel
  newhOverEEndcap
  hcalTowerSumEtConeDR04Barrel
  hcalTowerSumEtConeDR04Endcap
  hcalTowerBcSumEtConeDR04Barrel
  hcalTowerBcSumEtConeDR04Endcap


EOF


cat > unscaledhistosForPhotons <<EOF
pEResVsR9All
pEResVsR9Barrel
pEResVsR9Endcap
scpEResVsR9All
scpEResVsR9Barrel
scpEResVsR9Endcap
pEResVsEtAll
pEResVsEtBarrel
pEResVsEtEndcap
pEResVsEtaAll
pEResVsEtaUnconv
pEResVsEtaConv
pEcalRecHitSumEtConeDR04VsEtaAll
pEcalRecHitSumEtConeDR04VsEtBarrel
pEcalRecHitSumEtConeDR04VsEtEndcap
pHcalTowerSumEtConeDR04VsEtaAll
pHcalTowerSumEtConeDR04VsEtBarrel
pHcalTowerSumEtConeDR04VsEtEndcap
pHcalTowerBcSumEtConeDR04VsEtaAll
pHcalTowerBcSumEtConeDR04VsEtBarrel
pHcalTowerBcSumEtConeDR04VsEtEndcap
pHoverEVsEtaAll
pHoverEVsEtAll
pnewHoverEVsEtaAll
pnewHoverEVsEtAll


EOF



cat > 2dhistosForPhotons <<EOF
  R9VsEtaAll
  R1VsEtaAll
  R2VsEtaAll
  sigmaIetaIetaVsEtaAll
  isoTrkSolidConeDR04VsEtaAll
  nTrkSolidConeDR04VsEtaAll
  R9VsEtAll
  R1VsEtAll
  R2VsEtAll
  sigmaIetaIetaVsEtAll
  isoTrkSolidConeDR04VsEtAll
  nTrkSolidConeDR04VsEtAll
  eResVsR9All
  eResVsR9Barrel
  eResVsR9Endcap
  sceResVsR9All
  sceResVsR9Barrel
  sceResVsR9Endcap

EOF



endif






#=================END CONFIGURATION=====================

if (-e validation.C) rm validation.C
touch validation.C
cat > begin.C <<EOF
{
TFile *file_old = TFile::Open("$FASTSIM");
TFile *file_new = TFile::Open("$FULLSIM");

EOF
cat begin.C >>& validation.C
rm begin.C

setenv N 1





foreach i (`cat efficiencyForPhotons`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
if ( $i==deadChVsEta ||  $i==deadChVsPhi ||  $i==deadChVsEt ) {
$i->SetMinimum(0.);
$i->SetMaximum(0.2);
} else if (  $i==recoEffVsEt ) {
$i->GetXaxis()->SetRangeUser(0.,200.);
} else {
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
//file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
int nBins = $i->GetNbinsX();
float xMin=$i->GetBinLowEdge(1);
float xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/PhotonValidator/Photons");
file_old->cd("$HISTOPATHNAME_Photons");

TH1F* hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
if ( $i==scEAll || $i==phoEAll ) {  
$i->GetYaxis()->SetRangeUser(0.,2000.);
}
$i->SetStats(0);
$i->SetMinimum(0.);
//if ( mnew > mold) 
// $i->SetMaximum(mnew+mnew*0.2);
//else 
//$i->SetMaximum(mold+mold*0.2);
//$i->SetMaximum(mold+mold*0.2);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
//$i->SetLineWidth(3);
$i->Draw();
//file_new->cd("DQMData/EgammaV/PhotonValidator/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
//$i->SetLineWidth(1);
$i->Scale(nold/nnew);
TH1F* hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->Draw("e1same");
c$i->cd(2);
TH1F* ratio=new  TH1F("ratio"," ",nBins,xMin,xMax);
ratio->Divide(hnew,hold);
for ( int i=1; i<=ratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
ratio->SetBinError(i, erro);
}
ratio->SetStats(0);
ratio->SetLineColor(1);
ratio->SetLineWidth(2);
ratio->Draw("e");
TLine *l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");
//TString gifName=TString("gifs/$i")+"_ratio.gif";
//c$i->SaveAs(gifName);
EOF
  setenv N `expr $N + 1`
end





foreach i (`cat scaledhistosForPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->SetLogy(1);
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
file_old->cd("$HISTOPATHNAME_Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel ||  $i==hcalTowerSumEtConeDR04Endcap  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat unscaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
if ( $i==pEcalRecHitSumEtConeDR04VsEtaAll   ) {  
$i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtBarrel ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.); 
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtaAll   ) 
{ $i->GetYaxis()->SetRangeUser(0.,0.3);
} else if ( $i==pHcalTowerBcSumEtConeDR04VsEtaAll   ) 
{ $i->GetYaxis()->SetRangeUser(0.,1.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtBarrel ||  $i==pHcalTowerBcSumEtConeDR04VsEtBarrel) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtEndcap  || $i==pHcalTowerBcSumEtConeDR04VsEtEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pHoverEVsEtaAll || $i==pnewHoverEVsEtaAll  ) 
{ $i->GetYaxis()->SetRangeUser(-0.05,0.05);
} else if ( $i==pHoverEVsEtAll ||  $i==pnewHoverEVsEtAll ) 
{ $i->GetYaxis()->SetRangeUser(-0.05,0.05);
} else  {
$i->SetMinimum(0.8);
$i->SetMaximum(1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat 2dhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
file_old->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(2);
$i->SetMarkerSize(0.2);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


setenv NTOT `expr $N - 1`
setenv N 1
while ( $N <= $NTOT )
  cat temp$N.C >>& validation.C
  rm temp$N.C
  setenv N `expr $N + 1`
end

cat > end.C <<EOF
}
EOF
cat end.C >>& validation.C
rm end.C


if ( $TYPE == PixelMatchGsfElectron ) then
  setenv ANALYZER PixelMatchGsfElectronAnalyzer
  setenv CFG read_gsfElectrons
else if ( $TYPE == Photons ) then
  setenv ANALYZER PhotonValidator
  setenv CFG PhotonValidator_cfg
endif

if (-e validation.html) rm validation.html
if (-e validationPlotsTemplate.html) rm validationPlotsTemplate.html
cp ${CURRENTDIR}/validationPlotsTemplate.html validationPlotsTemplate.html
touch validation.html


cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$RELEASE : Full Sim vs Fast Sim $TYPE validation</title>
</head>

<h1>$RELEASE : Full Sim vs Fast Sim $TYPE validation
<br>
 $SAMPLE 
</h1>

<p>The following plots were made using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/src/$ANALYZER.cc">Validation/RecoEgamma/src/$ANALYZER</a>, 
using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/test/$CFG.py">Validation/RecoEgamma/test/$CFG.py</a>
<p>The script used to make the plots is <a href="validation.C">here</a>.
<br>
In all plots below, FastSim is in purple, Full Sim  in black.<br>
WARNING: no plots are not yet available for ecal seeded conversions in FastSim.
<br>
Click on the plots to see them enlarged.
<br>
Responsible: N. Marinelli
<br>
<br>

EOF


cat begin.html >>& validation.html
rm begin.html
cat  validationPlotsTemplate.html >>& validation.html
rm  validationPlotsTemplate.html 


rm scaledhistosForPhotons
rm unscaledhistosForPhotons
rm 2dhistosForPhotons
rm efficiencyForPhotons
rm scaledhistosForPhotonsLogScale

#echo "Now paste the following into your terminal window:"
#echo ""
echo "cd $OUTDIR"
#echo " root -b"
#echo ".x validation.C"
#echo ".q"
#echo "cd $CURRENTDIR"
#echo ""


root -b -l -q validation.C
cd $CURRENTDIR
echo "Then you can view your valdation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/$OUTPATH/validation.html"
