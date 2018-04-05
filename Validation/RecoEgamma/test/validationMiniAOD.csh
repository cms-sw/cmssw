##!/bin/csh -x
#!/bin/csh 

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  
#  #  "Validation/RecoEgamma/test/PhotonValidator_cfg.py
#
# The default list of histograms (configurable) is based on version VXX-XX-XX
# of Validation/RecoEgamma
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================

setenv RUNTYPE Central
#setenv RUNTYPE Local
setenv STARTUP True
setenv FASTSIM False
setenv UPGRADE True
## TYPE options: Photons, GEDPhotons
## ANALYZERNAME options: PhotonValidator, oldpfPhotonValidator, pfPhotonValidator
#setenv TYPE Photons
#setenv ANALYZERNAME PhotonValidator
setenv TYPE GEDPhotons
setenv ANALYZERNAME1 PhotonValidatorMiniAOD
setenv ANALYZERNAME2 pfPhotonValidator


setenv CMSSWver1 7_6_0
setenv CMSSWver2 7_6_0
setenv OLDRELEASE 7_6_0
setenv NEWRELEASE 7_6_0
setenv OLDPRERELEASE pre5
setenv NEWPRERELEASE pre5
setenv UPGRADEVER  UPG2017
setenv LHCENERGY   13
setenv PU False
setenv PUlevel 25ns

if ( $STARTUP == True &&  $FASTSIM == False && $PU == False && $UPGRADE == True ) then
setenv OLDGLOBALTAG MCRUN2_74_V7-v1
setenv NEWGLOBALTAG MCRUN2_74_V7-v2
else if ( $UPGRADE == True && $PU == True &&  $FASTSIM == False ) then
setenv OLDGLOBALTAG PU${PUlevel}_76X_mcRun2_asymptotic_v1-v1
setenv NEWGLOBALTAG PUpmx${PUlevel}_76X_mcRun2_asymptotic_v1-v1
else if (  $STARTUP == True  && $FASTSIM == True) then
setenv OLDGLOBALTAG MCRUN2_73_V7_FastSim-v1
setenv NEWGLOBALTAG MCRUN2_74_V1_FastSim-v1
 endif




setenv OLDRELEASE ${OLDRELEASE}_${OLDPRERELEASE}
#setenv OLDRELEASE ${OLDRELEASE}
setenv NEWRELEASE ${NEWRELEASE}_${NEWPRERELEASE}
#setenv NEWRELEASE ${NEWRELEASE}

setenv WorkDir1    /afs/cern.ch/user/n/nancy/scratch0/CMSSW/workForMINIAOD/CMSSW_7_6_0_pre5/src/Validation/RecoEgamma/myTest3

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

#Name of sample (affects output directory name and htmldescription only) 


#setenv SAMPLE SingleGammaPt10
#setenv SAMPLE SingleGammaPt35
##setenv SAMPLE SingleGammaFlatPt10_100
setenv SAMPLE H130GGgluonfusion
#setenv SAMPLE PhotonJets_Pt_10
#setenv SAMPLE GammaJets_Pt_80_120
#setenv SAMPLE QCD_Pt_80_120


if ( $RUNTYPE == Central ) then
setenv HISTOPATHNAME_Photons_miniAOD DQMData/Run\ 1/EgammaV/Run\ summary/${ANALYZERNAME1}/Photons
setenv HISTOPATHNAME_Photons DQMData/Run\ 1/EgammaV/Run\ summary/${ANALYZERNAME2}/Photons
endif

#==============END BASIC CONFIGURATION==================


#Input root trees for the two cases to be compared 

if ($SAMPLE == SingleGammaPt10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt10.root
else if ( $RUNTYPE == Central &&  $UPGRADE == False ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
else if ( $RUNTYPE == Central && $UPGRADE == True ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
#setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_${UPGRADEVER}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else 
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif



else if ($SAMPLE == SingleGammaPt35) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt35.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt35.root
else if ( $RUNTYPE == Central &&  $UPGRADE == False ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt35_UP15__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
else if ( $RUNTYPE == Central && $UPGRADE == True ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt35_UP15__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
#setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35_${UPGRADEVER}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else 
#setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif




else if ($SAMPLE == SingleGammaFlatPt10_100) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaFlatPt10To100.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaFlatPt10To100.root
endif 


else if ($SAMPLE == H130GGgluonfusion) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_H130GGgluonfusion.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_H130GGgluonfusion.root
else if ( $RUNTYPE == Central &&  $UPGRADE == False ) then


setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${UPGRADEVER}_${LHCENERGY}__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root

else if ( $RUNTYPE == Central && $UPGRADE == True ) then

#setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${LHCENERGY}__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQMIO.root
#setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${LHCENERGY}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQMIO.root
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root
setenv NEWFILE ${WorkDir1}/DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root
else
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQMIO.root
endif


else if ($SAMPLE == PhotonJets_Pt_10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_PhotonJets_Pt_10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_PhotonJets_Pt_10.root
else if ( $RUNTYPE == Central ) then

setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
if ( $UPGRADE == True ) then
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10_${UPGRADEVER}_${LHCENERGY}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif 
endif

else if ($SAMPLE ==  GammaJets_Pt_80_120) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_GammaJets_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_DQM.root
endif

else if ($SAMPLE == QCD_Pt_80_120) then 


endif



#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation/Photons
cd $OUTPATH
setenv NEWRELEASE {$NEWRELEASE}_MiniAOD
#setenv NEWRELEASE {$NEWRELEASE}
setenv OLDRELEASE {$OLDRELEASE}

if (! -d $NEWRELEASE) then
  mkdir $NEWRELEASE
endif
setenv OUTPATH $OUTPATH/$NEWRELEASE
cd $OUTPATH

if (! -d ${TYPE}) then
  mkdir ${TYPE}
endif
setenv OUTPATH $OUTPATH/${TYPE}
cd  $OUTPATH

if (! -d vs${OLDRELEASE}) then
  mkdir vs${OLDRELEASE}
endif
setenv OUTPATH $OUTPATH/vs${OLDRELEASE}


if ( $FASTSIM == True) then 
setenv OUTDIR $OUTPATH/${SAMPLE}FastSim
else if ( $FASTSIM == False && $UPGRADE == False && $PU == True ) then 
setenv OUTDIR $OUTPATH/${SAMPLE}PU
else if ( $FASTSIM == False && $PU == False && $UPGRADE == False ) then 
setenv OUTDIR $OUTPATH/${SAMPLE}
else if ( $SAMPLE == H130GGgluonfusion  && $UPGRADE == True && $PU == True && $FASTSIM == False) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV_PU${PUlevel}
else if ( $SAMPLE == H130GGgluonfusion  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE ==  PhotonJets_Pt_10  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE ==  SingleGammaPt10  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}
else if ( $SAMPLE ==  SingleGammaPt35  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}
endif


#else if ( $SAMPLE == H130GGgluonfusion ||  PhotonJets_Pt_10  && $UPGRADE == True ) then
#if ( $PU == True) then
#setenv OUTDIR $OUTPATH/${SAMPLE}PU
#else if ( $PU == False) then
#setenv OUTDIR $OUTPATH/${SAMPLE}
#endif



if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR


#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == Photons ||  $TYPE == GEDPhotons ) then


cat > efficiencyForPhotons <<EOF

EOF
#  recoEffVsEta
#  recoEffVsPhi
#  recoEffVsEt
#  deadChVsEta
#  deadChVsPhi
#  deadChVsEt



cat > scaledhistosForPhotons <<EOF
  scEta_miniAOD
  scPhi_miniAOD
  phoEAll_miniAOD
  phoEtAll_miniAOD
  eResAll_miniAOD
  eResBarrel_miniAOD
  eResEndcap_miniAOD
  sigmaEoEBarrel_miniAOD
  sigmaEoEEndcap_miniAOD
  isoTrkSolidConeDR04All_miniAOD
  isoTrkSolidConeDR04Barrel_miniAOD
  isoTrkSolidConeDR04Endcap_miniAOD
  nTrkSolidConeDR04All_miniAOD
  nTrkSolidConeDR04Barrel_miniAOD
  nTrkSolidConeDR04Endcap_miniAOD
  r9Barrel_miniAOD
  r9Endcap_miniAOD
  full5x5_r9Barrel_miniAOD
  full5x5_r9Endcap_miniAOD
  r1Barrel_miniAOD
  r1Endcap_miniAOD
  r2Barrel_miniAOD
  r2Endcap_miniAOD
  sigmaIetaIetaBarrel_miniAOD
  sigmaIetaIetaEndcap_miniAOD
  full5x5_sigmaIetaIetaBarrel_miniAOD
  full5x5_sigmaIetaIetaEndcap_miniAOD
  hOverEAll_miniAOD
  hOverEBarrel_miniAOD
  hOverEEndcap_miniAOD
  newhOverEAll_miniAOD
  newhOverEBarrel_miniAOD
  newhOverEEndcap_miniAOD
  hcalTowerSumEtConeDR04Barrel_miniAOD
  hcalTowerSumEtConeDR04Endcap_miniAOD
  hcalTowerBcSumEtConeDR04Barrel_miniAOD
  hcalTowerBcSumEtConeDR04Endcap_miniAOD
  ecalRecHitSumEtConeDR04Barrel_miniAOD
  ecalRecHitSumEtConeDR04Endcap_miniAOD


EOF

cat > scaledhistosForPhotonsLogScale <<EOF
  hOverEAll_miniAOD
  hOverEBarrel_miniAOD
  hOverEEndcap_miniAOD
  newhOverEAll_miniAOD
  newhOverEBarrel_miniAOD
  newhOverEEndcap_miniAOD
  hcalTowerSumEtConeDR04Barrel_miniAOD
  hcalTowerSumEtConeDR04Endcap_miniAOD
  hcalTowerBcSumEtConeDR04Barrel_miniAOD
  hcalTowerBcSumEtConeDR04Endcap_miniAOD
  ecalRecHitSumEtConeDR04Barrel_miniAOD
  ecalRecHitSumEtConeDR04Endcap_miniAOD
  r9Barrel_miniAOD
  r9Endcap_miniAOD
  full5x5_r9Barrel_miniAOD
  full5x5_r9Endcap_miniAOD
  r1Barrel_miniAOD
  r1Endcap_miniAOD
  r2Barrel_miniAOD
  r2Endcap_miniAOD
  sigmaIetaIetaBarrel_miniAOD
  sigmaIetaIetaEndcap_miniAOD



EOF




cat > scaledhistosGEDspecific <<EOF
  chargedHadIsoBarrel_miniAOD
  chargedHadIsoEndcap_miniAOD
  neutralHadIsoBarrel_miniAOD
  neutralHadIsoEndcap_miniAOD
  photonIsoBarrel_miniAOD
  photonIsoEndcap_miniAOD

EOF


cat > scaledhistosGEDspecificLogScale <<EOF
  photonIsoBarrel_miniAOD
  photonIsoEndcap_miniAOD
  chargedHadIsoBarrel_miniAOD
  chargedHadIsoEndcap_miniAOD
  neutralHadIsoBarrel_miniAOD
  neutralHadIsoEndcap_miniAOD


EOF




endif

#=================END CONFIGURATION=====================

if (-e validation.C) rm validation.C
touch validation.C
cat > begin.C <<EOF
{
TFile *file_old = TFile::Open("$OLDFILE");
TFile *file_new = TFile::Open("$NEWFILE");

int nBins = 0;
float xMin= 0;
float xMax= 0;
TH1F* hold;
TH1F* hnew;
TH1F* hratio;
TLine *l;
Double_t nnew;
Double_t mnew;
Double_t mold;
Double_t nold;
int scolor=kBlack;
int rcolor=kPink+8;
Double_t value;

EOF
cat begin.C >>& validation.C
rm begin.C

setenv N 1



foreach i (`cat scaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nBins = $i->GetNbinsX();
xMin=$i->GetBinLowEdge(1);
xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
mnew=$i->GetMaximum();
nnew=$i->GetEntries();
file_old->cd("$HISTOPATHNAME_Photons");
hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
mold=$i->GetMaximum();
nold=$i->GetEntries();
if ( $i==phoEAll_miniAOD ) {  
$i->GetYaxis()->SetRangeUser(0.,2000.);
} else if ( $i==r9Barrel_miniAOD ) {
$i->GetXaxis()->SetRangeUser(0.8,1.);
$i->GetYaxis()->SetRangeUser(0.,3500);
} else if ( $i==r9Endcap_miniAOD ) {
$i->GetXaxis()->SetRangeUser(0.8,1.);
$i->GetYaxis()->SetRangeUser(0.,1000);
} else if ( $i==sigmaIetaIetaBarrel_miniAOD ) {
$i->GetXaxis()->SetRangeUser(0.,0.02);
}  else if ( $i==sigmaIetaIetaEndcap_miniAOD ) {
$i->GetXaxis()->SetRangeUser(0.,0.06);
}  else if ( $i==r2Barrel_miniAOD || $i==r2Endcap_miniAOD) {
$i->GetXaxis()->SetRangeUser(0.7,1.02);
} 
$i->SetStats(1111);
$i->SetMinimum(0.);
$i->Draw();
gPad->Update();
TPaveStats *os$i = (TPaveStats*)$i->FindObject("stats");
os$i->SetX1NDC(0.82);
os$i->SetX2NDC(0.92);
os$i->SetY1NDC(0.8);
os$i->SetY2NDC(0.97);
os$i->SetTextColor(rcolor);
os$i->SetLineColor(rcolor);
if ( mnew > mold+sqrt(mold) )  { 
value=mnew+2*sqrt(mnew);
$i->SetMaximum(value); 
}  else { 
value=mold+2*sqrt(mold);
$i->SetMaximum(value); 
}
$i->SetLineColor(rcolor);
$i->SetFillColor(rcolor);
//$i->SetLineWidth(3);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
$i->SetStats(1111);
$i->SetLineColor(scolor);
$i->SetMarkerColor(scolor);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(0.8);
hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->Draw("e1sames");
gPad->Update();
TPaveStats *s$i = (TPaveStats*)$i->FindObject("stats");
s$i->SetX1NDC(0.82);
s$i->SetX2NDC(0.92);
s$i->SetY1NDC(0.55);
s$i->SetY2NDC(0.72);
c$i->cd(2);
hratio=new  TH1F("hratio"," ",nBins,xMin,xMax);
hratio->Divide(hnew,hold);
for ( int i=1; i<=hratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
hratio->SetBinError(i, erro);
}
hratio->SetStats(0);
hratio->SetLineColor(1);
hratio->SetLineWidth(2);
if ( $i==r9Barrel_miniAOD || $i==r9Endcap_miniAOD ) {
hratio->GetXaxis()->SetRangeUser(0.8,1.);
hratio->GetYaxis()->SetRangeUser(0.,2.);
} else if ( $i==eResAll_miniAOD || $i==eResBarrel_miniAOD || $i==eResEndcap_miniAOD ) {
hratio->GetYaxis()->SetRangeUser(0.,2.); 
} else if ( $i==sigmaIetaIetaBarrel_miniAOD ) {
hratio->GetXaxis()->SetRangeUser(0.,0.02);
}  else if ( $i==sigmaIetaIetaEndcap_miniAOD ) {
hratio->GetXaxis()->SetRangeUser(0.,0.06);
}  else if ( $i==r2Barrel_miniAOD || $i==r2Endcap_miniAOD) {
hratio->GetXaxis()->SetRangeUser(0.7,1.02);
hratio->GetYaxis()->SetRangeUser(0.,5.);
} else {
hratio->GetYaxis()->SetRangeUser(0.,2.);
}
hratio->Draw("e");
l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");
EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosGEDspecific`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nBins =$i->GetNbinsX();
xMin=$i->GetBinLowEdge(1);
xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
mnew=$i->GetMaximum();
nnew=$i->GetEntries();
file_old->cd("$HISTOPATHNAME_Photons");
hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
mold=$i->GetMaximum();
nold=$i->GetEntries();
if ( $i==phoEAll_miniAOD ) {  
$i->GetYaxis()->SetRangeUser(0.,2000.);
}
if ($i==chargedHadIsoBarrel_miniAOD || $i==chargedHadIsoBarrel_miniAOD ) {
$i->GetXaxis()->SetRangeUser(0.,12.);
}
$i->SetStats(11111);
$i->SetMinimum(0.);
$i->Draw();
gPad->Update();
TPaveStats *os2$i = (TPaveStats*)$i->FindObject("stats");
os2$i->SetTextColor(rcolor);
os2$i->SetLineColor(rcolor);

if ( mnew > mold+sqrt(mold) )  {
value=mnew+2*sqrt(mnew); 
$i->SetMaximum(value); 
}  else { 
value=mold+2*sqrt(mold);
$i->SetMaximum(value); 
}
$i->SetLineColor(rcolor);
$i->SetFillColor(rcolor);
$i->Draw();
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
$i->SetStats(111111);
$i->SetLineColor(scolor);
$i->SetMarkerColor(scolor);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(0.8);
hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->Draw("e1sames");
gPad->Update();
TPaveStats *s2$i = (TPaveStats*)$i->FindObject("stats");
s2$i->SetY1NDC(.5);
s2$i->SetY2NDC(.67);
c$i->cd(2);
hratio=new  TH1F("hratio"," ",nBins,xMin,xMax);
hratio->Divide(hnew,hold);
for ( int i=1; i<=hratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
hratio->SetBinError(i, erro);
}
hratio->SetStats(0);
hratio->SetLineColor(1);
hratio->SetLineWidth(2);
hratio->SetMinimum(0.);
hratio->SetMaximum(4.);
hratio->Draw("e");
l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");
EOF
  setenv N `expr $N + 1`
end



foreach i (`cat scaledhistosGEDspecificLogScale`)
  cat > temp$N.C <<EOF
TCanvas *cc$i = new TCanvas("cc$i");
cc$i->cd();
cc$i->SetFillColor(10);
cc$i->SetLogy();
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
file_old->cd("$HISTOPATHNAME_Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel_miniAOD ||  $i==hcalTowerSumEtConeDR04Endcap_miniAOD  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
nold=$i->GetEntries();
$i->SetStats(1111);
$i->SetMinimum(1);
$i->SetLineColor(rcolor);
$i->SetFillColor(rcolor);
$i->Draw();
gPad->Update();
TPaveStats *os3$i = (TPaveStats*)$i->FindObject("stats");
os3$i->SetTextColor(rcolor);
os3$i->SetLineColor(rcolor);

file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
$i->SetStats(11111);
$i->SetLineColor(scolor);
$i->SetMarkerColor(scolor);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Draw("e1sames");
gPad->Update();
TPaveStats *s3$i = (TPaveStats*)$i->FindObject("stats");
s3$i->SetY1NDC(.5);
s3$i->SetY2NDC(.67);
cc$i->SaveAs("gifs/log$i.gif");
EOF
  setenv N `expr $N + 1`
end





foreach i (`cat scaledhistosForPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *cc$i = new TCanvas("cc$i");
cc$i->cd();
cc$i->SetFillColor(10);
cc$i->SetLogy();
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
file_old->cd("$HISTOPATHNAME_Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel_miniAOD ||  $i==hcalTowerSumEtConeDR04Endcap_miniAOD  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
nold=$i->GetEntries();
$i->SetStats(1111);
$i->SetMinimum(1);
$i->SetLineColor(rcolor);
$i->SetFillColor(rcolor);
$i->Draw();
gPad->Update();
TPaveStats *os4$i = (TPaveStats*)$i->FindObject("stats");
os4$i->SetTextColor(rcolor);
os4$i->SetLineColor(rcolor);
file_new->cd("$HISTOPATHNAME_Photons_miniAOD");
nnew=$i->GetEntries();
$i->SetStats(11111);
$i->SetLineColor(scolor);
$i->SetMarkerColor(scolor);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Draw("e1sames");
gPad->Update();
TPaveStats *s4$i = (TPaveStats*)$i->FindObject("stats");
s4$i->SetY1NDC(.5);
s4$i->SetY2NDC(.67);
cc$i->SaveAs("gifs/log$i.gif");
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


if  ( $PU == True &&  $FASTSIM == False && $SAMPLE !=  H130GGgluonfusion  && $UPGRADE == True ) then
#setenv SAMPLE ${SAMPLE}PU
else if ( $PU == False && $FASTSIM == True) then
setenv SAMPLE ${SAMPLE}FastSim
else if ( $SAMPLE == H130GGgluonfusion && $UPGRADE == True && $PU == False  &&  $FASTSIM == False) then
setenv SAMPLE ${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE == H130GGgluonfusion && $UPGRADE == True &&  $PU == True  &&  $FASTSIM == False ) then
setenv SAMPLE ${SAMPLE}_${LHCENERGY}TeV_PU${PUlevel}
endif

echo $SAMPLE

if (-e validation.html) rm validation.html
if (-e validationPlotsTemplateForMiniAOD.html) rm validationPlotsTemplateForMiniAOD.html
cp ${CURRENTDIR}/validationPlotsTemplateForMiniAOD.html validationPlotsTemplateForMiniAOD.html
touch validation.html
cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$NEWRELEASE vs $OLDRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE vs $OLDRELEASE $TYPE validation
<br>
 $SAMPLE 
</h1>

<br>
In all plots below, $OLDRELEASE is in purple , $NEWRELEASE in black. 
<br>
Click on the plots to see them enlarged.
<br>
Responsible: N. Marinelli
<br>
<br>



EOF
cat begin.html >>& validation.html
rm begin.html
cat  validationPlotsTemplateForMiniAOD.html >>& validation.html
rm   validationPlotsTemplateForMiniAOD.html

rm efficiencyForPhotons
rm scaledhistosForPhotons
rm scaledhistosForPhotonsLogScale
rm scaledhistosGEDspecific
rm scaledhistosGEDspecificLogScale


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
echo "Then you can view your validation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/$OUTPATH/validation.html"
