#!/bin/bash

if [ -z $1 ]; then
  echo "$0 <benchmark> <loop>"
  exit
fi
if [ -z $1 ]; then
  echo "$0 <benchmark> <loop>"
  exit
fi
if [ ! -e "../$1" ]; then
  echo "Benchmark \"$1\" does not exist. Please build before running this script."
  exit
fi

script_dir=$(pwd)
bench_dir=$script_dir/../$1
i=0
fini=0

# compile
cd $bench_dir
make clean
make

# warmup
for t in {1..3} ; do
    fta=$(./run)
done

while true; do
    tstart=$(date +%s%N)

    for ft in {1..5} ; do
        fta=$(./run)

        i=$(( $i + 1 ))
        if [ $i -ge $2 ]; then
            fini=1
            break
        fi
    done

    tend=$((($(date +%s%N) - $tstart)/1000000))
    avg=$( echo "scale=3; ${tend} / ${ft}" | bc )
    echo "Average $avg ms per run"

    if [ $fini -eq 1 ] ; then
        break
    fi
done

cd $script_dir
