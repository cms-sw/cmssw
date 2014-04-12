#! /bin/tcsh 

## argv[1] == 1 
# copy reference files from afs and prepare cfg files

## argv[1] == 2
# submit cmsRun jobs


## argv[1] == 3 
# run the root macro
# copy plots and root files to afs

## argv[1] == 4 
# remove all input, intermediate and result files


### Hints ####################################################################
#
# to validate the tracks already present in the samples, use
#   set Sequence="only_validation"
#   set Tracks="generalTracks"
#
# to re-run the tracking and validate the new tracks, use
#   set Sequence="re_tracking"
#   set Tracks="cutsRecoTracks"
#

### Configuration ############################################################
set RefRelease="CMSSW_3_1_0_pre5"
set NewRelease="$CMSSW_VERSION"


# you 
#set samples=("RelValSingleMuPt1" "RelValSingleMuPt10" "RelValSingleMuPt100" "RelValSinglePiPt1" "RelValSinglePiPt10" "RelValSinglePiPt100")
set samples=("RelValSingleMuPt10" )
set Algo=""
#set Quality=""
set Quality="highPurity"
set Tracks="cutsRecoTracks"
#set Sequence="re_tracking"
#set Tracks="generalTracks"
set Sequence="only_validation"
set GlobalTag="IDEAL_30X"
set RefSelection="IDEAL_30X_noPU_ootb"
set NewSelection="${GlobalTag}_noPU_taula_highPuritytest"
set RefRepository="/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance"
set NewRepository="/afs/cern.ch/cms/performance/tracker/activities/reconstruction/tracking_performance"
set cfg="trackingPerformanceValidation_cfg.py"

##############################################################################
if ($1 == 1) then
echo "you chose option 1"

foreach sample($samples)

    mkdir -p $RefRelease
    if(! -e ${sample}_cff.py) then 
      echo "Missing "$sample"_cff.py, skipping this sample..."
      continue
    endif
    cp $RefRepository/$RefRelease/$RefSelection/$sample/val.$sample.root $RefRelease

    switch ($sample)
    case "RelValZPrimeEEM4000":
      set Events=1000
      breaksw
    case "RelValQCD_Pt_3000_3500":
      set Events=5000
      breaksw
    case "RelValTTbar":
      set Events=5000
      breaksw
    case "RelValSingleMu*":
      set Events=-1
      breaksw
    case "RelValSinglePi*":
      set Events=-1
      breaksw
    default:
      set Events=-1
    endsw
      
    #cp $RefRepository/$RefRelease/$RefSelection/$sample/${sample}_cff.py .
    cat ${sample}_cff.py | grep -v maxEvents >! tmp.py
    cat $cfg | sed \
      -e "s/NEVENT/$Events/g" \
      -e "s/GLOBALTAG/$GlobalTag/g" \
      -e "s/SEQUENCE/$Sequence/g" \
      -e "s/SAMPLE/$sample/g" \
      -e "s/TRACKS/$Tracks/g" \
      -e "s/[ ]\+source/source/g" \
    >> tmp.py
    if ($Algo != '') then
       cat tmp.py | sed \
       -e "s/ALGORITHM/'$Algo'/g" \
       >! tmp2.py
    else
       cat tmp.py | sed \
       -e "s/ALGORITHM//g" \
       >! tmp2.py
    endif
    if ($Quality != '') then 
	cat tmp2.py | sed \
	-e "s/QUALITY/'$Quality'/g" \
	>! $sample.py
     else
	cat tmp2.py | sed \
	-e "s/QUALITY//g" \
	>! $sample.py
    endif
rm tmp.py tmp2.py
end

##############################################################################
else if ($1 == 2) then
echo "you chose option 2"

eval `scramv1 run -csh`

foreach sample($samples)

  cmsRun $sample.py >& ! $sample.log < /dev/zero &

end

##############################################################################
else if ($1 == 3) then
echo "you chose option 3"
foreach sample($samples)

    cat macro/TrackValHistoPublisher.C | sed \
      -e s@NEW_FILE@val.$sample.root@g \
      -e s@REF_FILE@$RefRelease/val.$sample.root@g \
      -e s@REF_LABEL@$sample@g \
      -e s@NEW_LABEL@$sample@g \
      -e s@REF_RELEASE@$RefRelease@g \
      -e s@NEW_RELEASE@$NewRelease@g \
      -e s@REFSELECTION@$RefSelection@g \
      -e s@NEWSELECTION@$NewSelection@g \
      -e s@TrackValHistoPublisher@$sample@g \
      -e s@MINEFF@0.5@g \
      -e s@MAXEFF@1.025@g \
      -e s@MAXFAKE@0.7@g \
    > ! $sample.C

    root -b -q -l $sample.C > ! macro.$sample.log
    #$CMSSW_RELEASE_BASE/test/slc4_ia32_gcc345/hltTimingSummary -t 10000 -b100 -o $sample.timing -s -i output.$sample.root
    mkdir -p $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying pdf files for sample: " $sample
    cp *.pdf $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying root file for sample: " $sample
    cp val.$sample.root $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cff file for sample: " $sample
    cp ${sample}_cff.py $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cfg file for sample: " $sample
    cp $sample.py $NewRepository/$NewRelease/$NewSelection/$sample

end




##############################################################################
else if ($1 == 4) then
echo "you chose option 4"
foreach sample( $samples)
    rm -f $sample.cfg
    rm -f ${sample}_cff.py  
    rm -f $sample.C
    rm -f $sample.log
    rm -f val.$sample.root
    rm -f macro.$sample.log
end

else

    echo "you have to choose among option 1, option 2, option 3 and option 4"

endif
