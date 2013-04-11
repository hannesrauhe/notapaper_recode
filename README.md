notapaper_recode
================

You need CUDA and boost::random to compile this...

After compilation:

./maprecode -h

The output is Semicolon separated:
*Size of Dict in KB
*Throughput MB/s CPU sequential
*Throughput MB/s CPU openmp parallelized
*Throughput MB/s GPU transfer only
*Throughput MB/s GPU streaming without initial map-transfer and without processing
*Throughput MB/s GPU transfer,streaming and processing <- the Total

See http://blog.notapaper.de/article4.html
