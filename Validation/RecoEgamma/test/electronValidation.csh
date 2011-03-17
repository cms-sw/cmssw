#!/bin/csh

# This script can be used to generate a web page to compare histograms from 
# two input root files produced using the EDAnalyzers

#============= Configuration =================
# This script behavior is tuned by few unix variables and command-line
# arguments. You can use Oval as execution manager, whose configuration is
# within OvalFile. Oval will set the relevant variables for you.
# If you prefer to set the variables and run this script directly,
# here they are :
#
# $1 : eventual first command-line argument, immediatly duplicated into VAL_ENV,
#   is the name of the current context, used to build some default value
#   for other variables, especially VAL_NEW_FILE and VAL_REF_FILE.
# $2 : eventual second command-line argument, immediatly duplicated into VAL_OUTPUT_FILE,
#   is the default base name of the files containing the histograms ; it is
#   also used to build some default value for other variables.
# $3 : eventual third command-line argument, immediatly duplicated into VAL_WEB_SUB_DIR,
#   it is the name of the web subdirectory. Default is close to ${DBS_SAMPLE}_{DBS_COND}.
#
# VAL_COMMENT : a comment to inserted at the beginning of the web page, which generally
#   explains the specificity of the new release if this is not straightforward.
# VAL_NEW_RELEASE : chosen name for the new release to validate ; used in web pages
#   and used to build the path where the web pages will be stored.
# VAL_REF_RELEASE : chosen name of the old release to compare with ; used in web pages,
#   for default reference file path, and used to build the path where the web pages will
#   be stored.
#
# VAL_NEW_FILE : complete path of the file containing the new histograms.
#   If not set, a default value is computed, based on 1st command line argument,
#   2nd command line argument and current directory.
# VAL_REF_FILE : complete path of the file containing the old histograms to compare with.
#   If not set, a default value is computed, based on 1st command line argument,
#   2nd command line argument and VAL_REF_RELEASE.
# 
# VAL_HISTOS : name of the file describing the histograms to extract and generate.
# VAL_ANALYZER : name of the analyzer used.
# VAL_CONFIGURATION : name of the cmsRun python configuration used.
#
# DBS_SAMPLE : short chosen name for the current dataset ; used in web pages
#   and used to build the subdirectory where the web pages will be
#   stored ($VAL_WEB_SUB_DIR) unless it was given as the 3rd command line argument.
# DBS_COND : expression for the current conditions tag ; used to build the subdirectory
#   where the web pages will be stored ($VAL_WEB_SUB_DIR) unless it was given as the
#   3rd command line argument.
#=========================================================================================


#============== Core config ==================

setenv VAL_ENV $1
setenv VAL_OUTPUT_FILE $2
setenv VAL_WEB_SUB_DIR $3
setenv VAL_ORIGINAL_DIR $cwd

# those must have a value
#setenv VAL_WEB ...
#setenv VAL_WEB_URL ...
#setenv VAL_NEW_RELEASE ...
#setenv VAL_REF_RELEASE ...
# those either have a value, or will receive a default below
#setenv VAL_NEW_FILE ...
#setenv VAL_REF_FILE ...

#============== Find and prepare main output directory ==================

echo "VAL_WEB = ${VAL_WEB}"

if (! -d $VAL_WEB/$VAL_NEW_RELEASE) then
  mkdir $VAL_WEB/$VAL_NEW_RELEASE
endif

if (! -d $VAL_WEB/$VAL_NEW_RELEASE/Electrons) then
  mkdir $VAL_WEB/$VAL_NEW_RELEASE/Electrons
endif

if (! -d $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}) then
  mkdir $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}
endif

#============== Find and archive new log and data files ==================

echo "VAL_NEW_RELEASE = ${VAL_NEW_RELEASE}"

if ( ${?VAL_NEW_FILE} == "0" ) setenv VAL_NEW_FILE ""

if ( ${VAL_NEW_FILE} == "" ) then
  if ( -r "${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_NEW_FILE "${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}"
  endif
endif

if ( ${VAL_NEW_FILE} == "" ) then
  if ( -r "${VAL_WEB}/${VAL_NEW_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_NEW_FILE "${VAL_WEB}/${VAL_NEW_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}"
  endif
endif

if ( -r "${VAL_NEW_FILE}" ) then
  echo "VAL_NEW_FILE = ${VAL_NEW_FILE}"
else
  echo "${VAL_NEW_FILE} is unreadable !"
  setenv VAL_NEW_FILE ""
endif
  
#============== Find reference data file (eventually the freshly copied new data) ==================

echo "VAL_REF_RELEASE = ${VAL_REF_RELEASE}"

if ( ${?VAL_REF_FILE} == "0" ) setenv VAL_REF_FILE ""

if ( ${VAL_REF_FILE} == "" ) then
  if ( -r "${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/${VAL_NEW_FILE:t}" ) then
    setenv VAL_REF_FILE ${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/${VAL_NEW_FILE:t}
  endif
endif

if ( ${VAL_REF_FILE} == "" ) then
  if ( -r "${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_REF_FILE ${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}
  endif
endif

if ( ${VAL_REF_FILE} == "" ) then
  if ( -r "${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.gsfElectronHistos.root" ) then
    setenv VAL_REF_FILE ${VAL_WEB}/${VAL_REF_RELEASE}/Electrons/data/cmsRun.${VAL_ENV}.olog.gsfElectronHistos.root
  endif
endif

if ( ${VAL_REF_FILE} == "" ) then
  if ( -r "${VAL_WEB}/${VAL_REF_RELEASE}/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_REF_FILE ${VAL_WEB}/${VAL_REF_RELEASE}/data/cmsRun.${VAL_ENV}.olog.${VAL_OUTPUT_FILE}
  endif
endif

if ( ${VAL_REF_FILE} == "" ) then
  if ( -r "${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.oref.${VAL_OUTPUT_FILE}" ) then
    setenv VAL_REF_FILE ${VAL_ORIGINAL_DIR}/cmsRun.${VAL_ENV}.oref.${VAL_OUTPUT_FILE}
  endif
endif
 
echo "VAL_REF_FILE = ${VAL_REF_FILE}"
 
#============== Prepare sample/cond subdirectory ==================

if ( ${VAL_WEB_SUB_DIR} == "" ) then
  if ( "${DBS_COND}" =~ *MC* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Mc
  else if ( "${DBS_COND}" =~ *IDEAL* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Ideal
  else if ( "${DBS_COND}" =~ *START* ) then
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}_Startup
  else
    setenv VAL_WEB_SUB_DIR ${DBS_SAMPLE}
  endif
endif

if ( "${DBS_COND}" =~ *FastSim* ) then
      setenv VAL_WEB_SUB_DIR FastSim_${VAL_WEB_SUB_DIR}
endif

echo "VAL_WEB_SUB_DIR = ${VAL_WEB_SUB_DIR}"

if (! -d $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}) then
  mkdir $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}
endif

if (! -d $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/gifs) then
  mkdir $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/gifs
endif

cp -f ${VAL_ORIGINAL_DIR}/electronValidation.C $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}

#============== Prepare the list of histograms ==================
# The second argument is 1 if the histogram is scaled, 0 otherwise
# The third argument is 1 if the histogram is in log scale, 0 otherwise
# The fourth argument is 1 if the histogram is drawn with errors, 0 otherwise

cp $VAL_HISTOS $VAL_WEB/$VAL_NEW_RELEASE/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/histos.txt

#================= Generate the gifs and index.html =====================

root -b -l -q electronValidation.C
echo "You can access the files here:"
echo "${VAL_WEB}/${VAL_NEW_RELEASE}/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/"
echo "You can browse your validation plots here:"
echo "${VAL_WEB_URL}/${VAL_NEW_RELEASE}/Electrons/vs${VAL_REF_RELEASE}/${VAL_WEB_SUB_DIR}/"
