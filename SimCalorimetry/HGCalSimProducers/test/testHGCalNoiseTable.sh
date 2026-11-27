#!/bin/bash
# With the Phase-2 defaults the noise-only HGCAL digis of each event do not depend on the number of streams.

function die { echo "$1: status $2"; exit $2; }

function noiseDigest {
  name=$1
  shift
  cmsRun ${SCRAM_TEST_PATH}/hgcalNoiseTable_cfg.py digest=${name}.txt "$@" > ${name}.log 2>&1 || die "cmsRun hgcalNoiseTable_cfg.py $*" $?
  sort ${name}.txt > ${name}.sorted
}

noiseDigest noiseTable1 streams=1
noiseDigest noiseTable4 streams=4
# 8 events x 3 collections = 24 lines. EE and HEfront have noise digis in every event, HEback has
# none in these events: at least 2 x 8 = 16 lines with digis.
[ $(wc -l < noiseTable1.sorted) -eq 24 ] || die "counting the lines of noiseTable1.sorted" 1
[ $(awk '$5 > 0' noiseTable1.sorted | wc -l) -ge 16 ] || die "finding noise digis in noiseTable1.sorted" 1
diff noiseTable1.sorted noiseTable4.sorted || die "comparing the noise digis of 1 and 4 streams" $?

# Without noiseTableFromJobSeed each stream fills its table from its first event, so the digis depend on the stream.
noiseDigest noiseTableStream1 streams=1 noiseTableFromJobSeed=0
noiseDigest noiseTableStream4 streams=4 noiseTableFromJobSeed=0
cmp -s noiseTableStream1.sorted noiseTableStream4.sorted && die "the noise digis of 1 and 4 streams are equal without noiseTableFromJobSeed" 1

exit 0
