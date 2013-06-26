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

## argv[1] == 5
# run the comparision between CosmicTF/CTF/RS for NewRelease

### Hints ####################################################################
#
# to validate the tracks already present in the samples, use
#   set Sequence="only_validation"
#   #
# to re-run the tracking and validate the new tracks, use
#   set Sequence="re_tracking"
#
# you need to have an input file as RelValCosmic_cff.py!
#   #

### Configuration ############################################################
set RefRelease="CMSSW_3_2_5"
set NewRelease="$CMSSW_VERSION"

# you 
set samples=("RelValCosmic")
set Algo=""
set Quality=""
# For only re_tracking and only_validation are tested!!
set Sequence="re_tracking"
#"only_validation"
set GlobalTag="MC_31X_V5"
set RefSelection="MC_31X_V5_ootb"
set NewSelection="${GlobalTag}_ootb"
set RefRepository=""
set NewRepository=""
set cfg="cosmictrackingPerformanceValidation_cfg.py"

# Setting the plotting ranges
set mineta="-1.5"
set maxeta="1.5"
set ninteta="30"
set minpt="0.0"
set maxpt="50.0"
set nintpt="20"
set mindxy="-100.0"
set maxdxy="100.0"
set nintdxy="25"
set mindz="-300.0"
set maxdz="300.0"
set nintdz="25"
set minphi="-3.15"
set maxphi="0.0"
set nintphi="25"
set minvertpos="0.0"
set maxvertpos="100.0"
set nintvertpos="50"
set minzpos="-300.0"
set maxzpos="300.0"
set nintzpos="25"
set dxyresmin="-0.06"
set dxyresmax="0.06"
set dzresmin="-0.08"
set dzresmax="0.08"
set phiresmin="-0.0035"
set phiresmax="0.0035"
set ptresmin="-0.2"
set ptresmax="0.2"
set cotthetaresmin="-0.02"
set cotthetaresmax="0.02"

# Setting the cuts on TP
set pttpmin="1.0"
set liptpmax="300.0"
set tiptpmax="110.0"
set etatpmin="-2.0"
set etatpmax="2.0"
set nhittpmin="5"

if ($NewSelection == "${GlobalTag}_ootb" ) then 
  # Setting the cuts on RecoTracks
  set ptrecotrkmin="0.1"
  set liprecotrkmax="9999"
  set tiprecotrkmax="9999"
  set etarecotrkmin="-5.0"
  set etarecotrkmax="5.0"
  set nhitrecotrkmin="0" # number of crossed layers
  set chisqrecotrkmax="99999"
endif

if ($NewSelection == "${GlobalTag}_loose" ) then 
  # Setting the cuts on RecoTracks
  set ptrecotrkmin="1.0"
  set liprecotrkmax="300.0"
  set tiprecotrkmax="110.0"
  set etarecotrkmin="-2.0"
  set etarecotrkmax="2.0"
  set nhitrecotrkmin="5" # number of crossed layers
  set chisqrecotrkmax="10" 
endif

if ($NewSelection == "${GlobalTag}_tight" ) then 
  # set the plotting range
  set mineta="-1.5"
  set maxeta="1.5"
  set ninteta="24"
  set minpt="0.0"
  set maxpt="50.0"
  set nintpt="20"
  set mindxy="-100.0"
  set maxdxy="100.0"
  set nintdxy="50"
  set mindz="-100.0"
  set maxdz="100.0"
  set nintdz="50"
  set minphi="-3.15"
  set maxphi="0.0"
  set nintphi="50"
  set minvertpos="0.0"
  set maxvertpos="100.0"
  set nintvertpos="100"
  set minzpos="-100.0" 
  set maxzpos="100.0"
  set nintzpos="50"
  # Setting the cuts on TP
  set pttpmin="1.0"
  set liptpmax="100.0"
  set tiptpmax="100.0"
  set etatpmin="-1.5"
  set etatpmax="1.5"
  set nhittpmin="5" # number of crossed layers

  # Setting the cuts on RecoTracks
  set ptrecotrkmin="1.0"
  set liprecotrkmax="100"
  set tiprecotrkmax="100"
  set etarecotrkmin="-1.5"
  set etarecotrkmax="1.5"
  set nhitrecotrkmin="5" # number of crossed layers
  set chisqrecotrkmax="10"
endif

if ($NewSelection == "${GlobalTag}_tighter" ) then 
  # set the plotting range
  set mineta="-1"
  set maxeta="1"
  set ninteta="10"
  set minpt="0.0"
  set maxpt="50.0"
  set nintpt="10"
  set mindxy="-50.0"
  set maxdxy="50.0"
  set nintdxy="20"
  set mindz="-50.0"
  set maxdz="50.0"
  set nintdz="20"
  set minphi="-2.5"
  set maxphi="-0.5"
  set nintphi="20"
  set minvertpos="0.0"
  set maxvertpos="50.0"
  set nintvertpos="20"
  set minzpos="-50.0" 
  set maxzpos="50.0"
  set nintzpos="20"
  # Setting the cuts on TP
  set pttpmin="1.0"
  set liptpmax="50.0"
  set tiptpmax="50.0"
  set etatpmin="-1"
  set etatpmax="1"
  set nhittpmin="5"

  # Setting the cuts on RecoTracks
  set ptrecotrkmin="1.0"
  set liprecotrkmax="50"
  set tiprecotrkmax="50"
  set etarecotrkmin="-1"
  set etarecotrkmax="1"
  set nhitrecotrkmin="5" # number of crossed layers
  set chisqrecotrkmax="10"
endif


if ($NewSelection == "${GlobalTag}_lhc" ) then 
  # set the plotting range
  set mineta="-1.0"
  set maxeta="1.0"
  set ninteta="15"
  set minpt="0.0"
  set maxpt="50.0"
  set nintpt="50"
  set mindxy="-30.0"
  set maxdxy="30.0"
  set nintdxy="50"
  set mindz="-30.0"
  set maxdz="30.0"
  set nintdz="50"
  set minphi="-3.15"
  set maxphi="0.0"
  set nintphi="50"
  set minvertpos="0.0"
  set maxvertpos="30.0"
  set nintvertpos="50"
  set minzpos="-30.0" 
  set maxzpos="30.0"
  set nintzpos="50"
  # Setting the cuts on TP
  set pttpmin="1.0"
  set liptpmax="30.0"
  set tiptpmax="30.0"
  set etatpmin="-1.0"
  set etatpmax="1.0"
  set nhittpmin="5"

  # Setting the cuts on RecoTracks
  set ptrecotrkmin="1.0"
  set liprecotrkmax="30"
  set tiprecotrkmax="30"
  set etarecotrkmin="-1.0"
  set etarecotrkmax="1.0"
  set nhitrecotrkmin="5" # number of crossed layers
  set chisqrecotrkmax="10.0"
endif

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

    set Events=10000
      
    cp $RefRepository/$RefRelease/$RefSelection/$sample/${sample}_cff.py .
    cat ${sample}_cff.py | grep -v maxEvents >! tmp.py
    cat $cfg | sed \
      -e "s/NEVENT/$Events/g" \
      -e "s/GLOBALTAG/$GlobalTag/g" \
      -e "s/SEQUENCE/$Sequence/g" \
      -e "s/SAMPLE/$sample/g" \
      -e "s/MINETA/$mineta/g" \
      -e "s/MAXETA/$maxeta/g" \
      -e "s/NINTETA/$ninteta/g" \
      -e "s/MINPT/$minpt/g" \
      -e "s/MAXPT/$maxpt/g" \
      -e "s/NINTPT/$nintpt/g" \
      -e "s/MINDXY/$mindxy/g" \
      -e "s/MAXDXY/$maxdxy/g" \
      -e "s/NINTDXY/$nintdxy/g" \
      -e "s/MINDZ/$mindz/g" \
      -e "s/MAXDZ/$maxdz/g" \
      -e "s/NINTDZ/$nintdz/g" \
      -e "s/MINPHI/$minphi/g" \
      -e "s/MAXPHI/$maxphi/g" \
      -e "s/NINTPHI/$nintphi/g" \
      -e "s/MINVERTPOS/$minvertpos/g" \
      -e "s/MAXVERTPOS/$maxvertpos/g" \
      -e "s/NINTVERTPOS/$nintvertpos/g" \
      -e "s/MINZPOS/$minzpos/g" \
      -e "s/MAXZPOS/$maxzpos/g" \
      -e "s/NINTZPOS/$nintzpos/g" \
      -e "s/DXYRESMIN/$dxyresmin/g" \
      -e "s/DXYRESMAX/$dxyresmax/g" \
      -e "s/DZRESMIN/$dzresmin/g" \
      -e "s/DZRESMAX/$dzresmax/g" \
      -e "s/PHIRESMIN/$phiresmin/g" \
      -e "s/PHIRESMAX/$phiresmax/g" \
      -e "s/PTRESMIN/$ptresmin/g" \
      -e "s/PTRESMAX/$ptresmax/g" \
      -e "s/COTTHETARESMIN/$cotthetaresmin/g" \
      -e "s/COTTHETARESMAX/$cotthetaresmax/g" \
      -e "s/PTTPMIN/$pttpmin/g" \
      -e "s/LIPTPMAX/$liptpmax/g" \
      -e "s/TIPTPMAX/$tiptpmax/g" \
      -e "s/ETATPMIN/$etatpmin/g" \
      -e "s/ETATPMAX/$etatpmax/g" \
      -e "s/NHITTPMIN/$nhittpmin/g" \
      -e "s/PTRECOTRKMIN/$ptrecotrkmin/g" \
      -e "s/LIPRECOTRKMAX/$liprecotrkmax/g" \
      -e "s/TIPRECOTRKMAX/$tiprecotrkmax/g" \
      -e "s/ETARECOTRKMIN/$etarecotrkmin/g" \
      -e "s/ETARECOTRKMAX/$etarecotrkmax/g" \
      -e "s/NHITRECOTRKMIN/$nhitrecotrkmin/g" \
      -e "s/CHISQRECOTRKMAX/$chisqrecotrkmax/g" \
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

    mkdir -p CosmicTF
    mkdir -p CTF
    mkdir -p RS
    # set true if you want to have the eps file for each plot
    set ifploteps="false"

    cat macro/CosmicTrackValHistoPublisher.C | sed \
      -e s@NEW_FILE@val.$sample.root@g \
      -e s@REF_FILE@$RefRepository/$RefRelease/$RefSelection/$sample/val.$sample.root@g \
      -e s@REF_LABEL@$sample@g \
      -e s@NEW_LABEL@$sample@g \
      -e s@REF_RELEASE@$RefRelease@g \
      -e s@NEW_RELEASE@$NewRelease@g \
      -e s@REFSELECTION@$RefSelection@g \
      -e s@NEWSELECTION@$NewSelection@g \
      -e s@CosmicTrackValHistoPublisher@$sample@g \
      -e s@MINEFF@0.0@g \
      -e s@MAXEFF@1.025@g \
      -e s@MAXFAKE@0.2@g \
      -e s@IFPLOTEPS@$ifploteps@g \
    > ! $sample.C

    root -b -q -l $sample.C > ! macro.$sample.log

    mkdir -p $NewRepository/$NewRelease/$NewSelection/$sample
  
    echo "copying pdf files for sample: " $sample
    cp -r CosmicTF $NewRepository/$NewRelease/$NewSelection/$sample
    cp -r CTF $NewRepository/$NewRelease/$NewSelection/$sample
    cp -r RS $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying root file for sample: " $sample
    cp val.$sample.root $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cff file for sample: " $sample
    cp ${sample}_cff.py $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying cfg file for sample: " $sample
    cp $sample.py $NewRepository/$NewRelease/$NewSelection/$sample

    echo "copying macro file for sample: " $sample
    cp $sample.C $NewRepository/$NewRelease/$NewSelection/$sample

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
    rm -r cosmic_algoval_plots
    rm -r CosmicTF
    rm -r CTF
    rm -r RS
    rm -f AlgoValCosmic.C
end


##############################################################################
else if ($1 == 5) then
echo "you chose option 5"
foreach sample($samples)
   
    mkdir -p cosmic_algoval_plots
    
    # set true if you want to have the eps file for each plot
    set ifploteps="false"
   
    cat macro/AlgoValCosmic.C | sed \
      -e s@NEW_FILE@val.$sample.root@g \
      -e s@NEW_RELEASE@$NewRelease@g \
      -e s@NEWSELECTION@$NewSelection@g \
      -e s@MINEFF@0.0@g \
      -e s@MAXEFF@1.025@g \
      -e s@MAXFAKE@0.2@g \
      -e s@IFPLOTEPS@$ifploteps@g \
    > ! AlgoValCosmic.C

    root -b -q -l AlgoValCosmic.C > ! macro.AlgoValCosmic.log

    mkdir -p $NewRepository/$NewRelease/$NewSelection/$sample
  
    echo "copying cosmic algovalidation files for sample: " $sample
    cp -r cosmic_algoval_plots $NewRepository/$NewRelease/$NewSelection/$sample
   
    echo "copying cosmic algovalidation files for sample: " $sample
    cp  AlgoValCosmic.C $NewRepository/$NewRelease/$NewSelection/$sample


end



else

    echo "you have to choose among option 1, option 2, option 3, option 4 and option 5"

endif
