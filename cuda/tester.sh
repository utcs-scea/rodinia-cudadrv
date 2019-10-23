#!/usr/bin/env bash

source benchmarks

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

tt=0
i=0
e2etime=0

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    make clean && make
    for idx in `seq 1 15`; do
        tstart=$(date +%s%N)

        exe ./run

        tend=$((($(date +%s%N) - $tstart)/1000000))
        e2etime=$(( $tend + $e2etime ))
        i=$(( $i + 1 ))
        exe echo "$(date) # end2end elapsed $tend ms"

        exe echo
    done
    cd $DIR
    exe echo
    echo
done

et=$( echo "scale=3; $e2etime / $i " | bc )
echo "Average ${et}ms per run"
