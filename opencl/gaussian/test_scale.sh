#!/bin/bash

DIR=.
OUTDIR=$DIR
scale=4
b="scale_${scale}"

function exe() {
    echo "++ $@" | tee -a $OUTDIR/$b.txt
    "$@" |& tee -a $OUTDIR/$b.txt
}

function do_single_run() {
    seq $scale | xargs -P $scale -n 1 bash -c "$1 ./run"
}

export -f exe
export OUTDIR b

echo -n > $OUTDIR/$b.txt # clean output file
echo "$(date) # running $b"
#make clean && make
for idx in `seq 1 3`; do
    do_single_run
done

for idx in `seq 1 10`; do
    do_single_run exe
    exe echo
done
