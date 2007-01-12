#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if (-e /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/ ) mkdir /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/

setenv WWWDIRObj /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/SimHit

if (-e $WWWDIRObj) mkdir $WWWDIRObj

setenv WWWDIR $WWWDIRObj

mkdir $WWWDIR/eps
mkdir $WWWDir/eps/TK-strips
mkdir $WWWDir/eps/TK-strips/eloss
mkdir $WWWDir/eps/TK-strips/Localx
mkdir $WWWDir/eps/TK-strips/Localy
mkdir $WWWDir/eps/TK-strips/Entryx-Exitx
mkdir $WWWDir/eps/TK-strips/Entryy-Exity
mkdir $WWWDir/eps/TK-strips/Entryz-Exitz
mkdir $WWWDir/eps/TK-pixels
mkdir $WWWDir/eps/TK-pixels/eloss
mkdir $WWWDir/eps/TK-pixels/Localx
mkdir $WWWDir/eps/TK-pixels/Localy
mkdir $WWWDir/eps/TK-pixels/Entryx-Exitx
mkdir $WWWDir/eps/TK-pixels/Entryy-Exity
mkdir $WWWDir/eps/TK-pixels/Entryz-Exitz
mkdir $WWWDir/eps/TK-summary


mkdir $WWWDIR/gif
mkdir $WWWDir/gif/TK-strips
mkdir $WWWDir/gif/TK-strips/eloss
mkdir $WWWDir/gif/TK-strips/Localx
mkdir $WWWDir/gif/TK-strips/Localy
mkdir $WWWDir/gif/TK-strips/Entryx-Exitx
mkdir $WWWDir/gif/TK-strips/Entryy-Exity
mkdir $WWWDir/gif/TK-strips/Entryz-Exitz
mkdir $WWWDir/gif/TK-pixels
mkdir $WWWDir/gif/TK-pixels/eloss
mkdir $WWWDir/gif/TK-pixels/Localx
mkdir $WWWDir/gif/TK-pixels/Localy
mkdir $WWWDir/gif/TK-pixels/Entryx-Exitx
mkdir $WWWDir/gif/TK-pixels/Entryy-Exity
mkdir $WWWDir/gif/TK-pixels/Entryz-Exitz
mkdir $WWWDir/gif/TK-summary

echo "...Copying..."
cp plots/muon/eloss_T*_KS*.eps $WWWDir/eps/TK-strips/eloss
cp plots/muon/pos_Entryx-Exitx_T*_KS*.eps $WWWDir/eps/TK-strips/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_T*_KS*.eps $WWWDir/eps/TK-strips/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_T*_KS*.eps $WWWDir/eps/TK-strips/Entryz-Exitz
cp plots/muon/pos_Localy_T*_KS*.eps $WWWDir/eps/TK-strips/Localy
cp plots/muon/pos_Localx_T*_KS*.eps $WWWDir/eps/TK-strips/Localx

cp plots/muon/eloss_*PIX_KS*.eps $WWWDir/eps/TK-pixels/eloss
cp plots/muon/pos_Entryx-Exitx_*PIX_KS*.eps $WWWDir/eps/TK-pixels/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_*PIX_KS*.eps $WWWDir/eps/TK-pixels/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_*PIX_KS*.eps $WWWDir/eps/TK-pixels/Entryz-Exitz
cp plots/muon/pos_Localy_*PIX_KS*.eps $WWWDir/eps/TK-pixels/Localy
cp plots/muon/pos_Localx_*PIX_KS*.eps $WWWDir/eps/TK-pixels/Localx

cp plots/muon/summary*.eps $WWWDir/eps/TK-summary
cp LowKS*.eps $WWWDir/eps/TK-summary

cp plots/muon/eloss_T*_KS*.gif $WWWDir/gif/TK-strips/eloss
cp plots/muon/pos_Entryx-Exitx_T*_KS*.gif $WWWDir/gif/TK-strips/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_T*_KS*.gif $WWWDir/gif/TK-strips/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_T*_KS*.gif $WWWDir/gif/TK-strips/Entryz-Exitz
cp plots/muon/pos_Localy_T*_KS*.gif $WWWDir/gif/TK-strips/Localy
cp plots/muon/pos_Localx_T*_KS*.gif $WWWDir/gif/TK-strips/Localx

cp plots/muon/eloss_*PIX_KS*.gif $WWWDir/gif/TK-pixels/eloss
cp plots/muon/pos_Entryx-Exitx_*PIX_KS*.gif $WWWDir/gif/TK-pixels/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_*PIX_KS*.gif $WWWDir/gif/TK-pixels/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_*PIX_KS*.gif $WWWDir/gif/TK-pixels/Entryz-Exitz
cp plots/muon/pos_Localy_*PIX_KS*.gif $WWWDir/gif/TK-pixels/Localy
cp plots/muon/pos_Localx_*PIX_KS*.gif $WWWDir/gif/TK-pixels/Localx

cp plots/muon/summary*.gif $WWWDir/gif/TK-summary
cp LowKS*.gif $WWWDir/gif/TK-summary
echo "...Done..."
