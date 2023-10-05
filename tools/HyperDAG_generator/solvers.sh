executable = $1
A=$2
L=$3
U=$4

$1 spmv -output spmv_$A -inputs $A
$1 lltsolver -output lltsolver_$A -inputs $L
$1 alltsolver -output alltsolver_$A -inputs $A $L
$1 lusolver -output lusolver_$A -inputs $L $U
$1 alusolver -output alusolver_$A -inputs $A $L $U
