#! /bin/tcsh 

### Configuration ############################################################
#set RefRelease="r39_step01"
set RefRelease="r39_step01_allTEC"
set NewRelease="r39_step01_44xtrk"
#set RefFile="phase1363_mtv_muPU0.root"
set RefFile="phase144xtrkallTEC_mtv_muPU0.root"
set NewFile="phase144xtrk_mtv_muPU0.root"
#set RefReleaseNote="r39_step01-PU0_CMSSW_3_6_3"
set RefReleaseNote="r39_step01-PU0_CMSSW_4_2_3"
set NewReleaseNote="r39_step01-PU0_CMSSW_4_2_3"

# you 
set RefDatafiles="plots"
set NewDatafiles="plots"
set samples=("pu00")
set RefSelection="muon"
set NewSelection="muon"
set NewRepository="plots/mtv"

##############################################################################
echo "making comparison plots and copying them to $NewRepository"
foreach sample($samples)

echo "RefFile = $RefDatafiles/$RefFile"
echo "NewFile = $NewDatafiles/$NewFile"
    cat macro/HPIterTrackValHistoPublisher.C | sed \
      -e s@NEW_FILE@$NewDatafiles/$NewFile@g \
      -e s@REF_FILE@$RefDatafiles/$RefFile@g \
      -e s@REF_LABEL@$sample@g \
      -e s@NEW_LABEL@$sample@g \
      -e s@REF_RELEASE@$RefReleaseNote@g \
      -e s@NEW_RELEASE@$NewReleaseNote@g \
      -e s@REFSELECTION@$RefSelection@g \
      -e s@NEWSELECTION@$NewSelection@g \
      -e s@HPIterTrackValHistoPublisher@$sample@g \
      -e s@MINEFF@0.0@g \
      -e s@MAXEFF@1.025@g \
      -e s@MAXFAKEETA@0.2@g \
      -e s@MAXFAKEPT@0.3@g \
      -e s@MAXFAKEHIT@0.5@g \
    > ! $sample.C

    root -b -q -l $sample.C > ! macro.$sample.log
    mkdir -p $NewRepository

    echo "copying pdf files for sample: " $sample
    cp *.pdf $NewRepository/

end

