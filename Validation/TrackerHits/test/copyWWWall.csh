#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/) mkdir /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/

setenv WWWDIR /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/SimHit

if ( ! -d $WWWDIR) mkdir $WWWDIR

mkdir $WWWDIR/eps
mkdir $WWWDIR/eps/TK-strips
mkdir $WWWDIR/eps/TK-strips/eloss
mkdir $WWWDIR/eps/TK-strips/Localx
mkdir $WWWDIR/eps/TK-strips/Localy
mkdir $WWWDIR/eps/TK-strips/Entryx-Exitx
mkdir $WWWDIR/eps/TK-strips/Entryy-Exity
mkdir $WWWDIR/eps/TK-strips/Entryz-Exitz
mkdir $WWWDIR/eps/TK-pixels
mkdir $WWWDIR/eps/TK-pixels/eloss
mkdir $WWWDIR/eps/TK-pixels/Localx
mkdir $WWWDIR/eps/TK-pixels/Localy
mkdir $WWWDIR/eps/TK-pixels/Entryx-Exitx
mkdir $WWWDIR/eps/TK-pixels/Entryy-Exity
mkdir $WWWDIR/eps/TK-pixels/Entryz-Exitz
mkdir $WWWDIR/eps/TK-summary

mkdir $WWWDIR/gif
mkdir $WWWDIR/gif/TK-strips
mkdir $WWWDIR/gif/TK-strips/eloss
mkdir $WWWDIR/gif/TK-strips/Localx
mkdir $WWWDIR/gif/TK-strips/Localy
mkdir $WWWDIR/gif/TK-strips/Entryx-Exitx
mkdir $WWWDIR/gif/TK-strips/Entryy-Exity
mkdir $WWWDIR/gif/TK-strips/Entryz-Exitz
mkdir $WWWDIR/gif/TK-pixels
mkdir $WWWDIR/gif/TK-pixels/eloss
mkdir $WWWDIR/gif/TK-pixels/Localx
mkdir $WWWDIR/gif/TK-pixels/Localy
mkdir $WWWDIR/gif/TK-pixels/Entryx-Exitx
mkdir $WWWDIR/gif/TK-pixels/Entryy-Exity
mkdir $WWWDIR/gif/TK-pixels/Entryz-Exitz
mkdir $WWWDIR/gif/TK-summary

echo "...Copying..."

cp plots/muon/eloss_T*_KS*.eps $WWWDIR/eps/TK-strips/eloss
cp plots/muon/pos_Entryx-Exitx_T*_KS*.eps $WWWDIR/eps/TK-strips/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_T*_KS*.eps $WWWDIR/eps/TK-strips/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_T*_KS*.eps $WWWDIR/eps/TK-strips/Entryz-Exitz
cp plots/muon/pos_Localy_T*_KS*.eps $WWWDIR/eps/TK-strips/Localy
cp plots/muon/pos_Localx_T*_KS*.eps $WWWDIR/eps/TK-strips/Localx

cp plots/muon/eloss_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/eloss
cp plots/muon/pos_Entryx-Exitx_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/Entryz-Exitz
cp plots/muon/pos_Localy_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/Localy
cp plots/muon/pos_Localx_*PIX_KS*.eps $WWWDIR/eps/TK-pixels/Localx

cp plots/muon/summary*.eps $WWWDIR/eps/TK-summary

cp plots/muon/eloss_T*_KS*.gif $WWWDIR/gif/TK-strips/eloss
cp plots/muon/pos_Entryx-Exitx_T*_KS*.gif $WWWDIR/gif/TK-strips/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_T*_KS*.gif $WWWDIR/gif/TK-strips/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_T*_KS*.gif $WWWDIR/gif/TK-strips/Entryz-Exitz
cp plots/muon/pos_Localy_T*_KS*.gif $WWWDIR/gif/TK-strips/Localy
cp plots/muon/pos_Localx_T*_KS*.gif $WWWDIR/gif/TK-strips/Localx

cp plots/muon/eloss_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/eloss
cp plots/muon/pos_Entryx-Exitx_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/Entryz-Exitz
cp plots/muon/pos_Localy_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/Localy
cp plots/muon/pos_Localx_*PIX_KS*.gif $WWWDIR/gif/TK-pixels/Localx

cp plots/muon/summary*.gif $WWWDIR/gif/TK-summary

echo "...Done..."
