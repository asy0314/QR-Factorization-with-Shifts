# QR-Factorization-with-Shifts
a: Almost upper triangular; b: Givens; c: Single-Shift vs. Wilkson shifts; d: Breaking symmetry

# How to run
1. install Julia version 1.8 on your computer
   
2. open your terminal, go to the directory of those files, and run the command: julia --project=. driver.jl
   
3. The implemetation will output two convergence figures in pdf at the current folder 

# contents of function_code.jl
a: reduce a symmetric matrix A to Hessenberg form using Householder reflections
operate in place, overwriting the input matrix with Hessenberg form

b: run a single iteration of the unshifted QR algorithm 
using Givens rotations to implement QR factorization on T_k
input = T_k hessenberg from 
output = T_{k+1} Hessenberg form

c: ren the practical QR iteration with both the Single-Shift and Wilkinson Shift. 
using QR iteration in b, with criteria for when to implement deflation and when to terminate QR iteration    

d: design an experiment that evaluates your practical QR algorithm with shifts
include a semi-log plot showing the rate of convergence of Single-Shift and Wilkinson Shift to compare
