import Pkg.instantiate
instantiate()
using BenchmarkTools: @ballocated
using LinearAlgebra: I, norm, istriu, triu, qr
using CairoMakie
include("HW4_your_code.jl")


#----------------------------------------
# Problem a 
#----------------------------------------
########################################
A = randn(30, 20) 
b = randn(30)
Q, R = classical_gram_schmidt(A) 
@assert Q' * Q ≈ I
@assert Q * R ≈ A

#----------------------------------------
# Problem b 
#----------------------------------------
########################################
A = randn(30, 20) 
b = randn(30)
Q, R = modified_gram_schmidt(A) 
@assert Q' * Q ≈ I
@assert Q * R ≈ A

#----------------------------------------
# Problem c
#----------------------------------------
########################################
A = randn(25, 20) 
allocated_memory = @ballocated  householder_QR!(A)
@assert allocated_memory == 0
A = randn(25, 20)
true_R = Matrix(qr(A).R)
householder_QR!(A)
# Checks if the R part of the factorization is correct
@assert vcat(true_R, zeros(5,20)) ≈ triu(A)

#----------------------------------------
# Problem d
#----------------------------------------
########################################
# Testing for memory allocation:
A = randn(25, 20) 
householder_QR!(A)
QR = A
x = randn(20)
b = randn(25)
out_mul = randn(25)
out_div = randn(20)

allocated_memory_mul = @ballocated  householder_QR_mul!(out_mul, x, QR)
allocated_memory_div = @ballocated  householder_QR_div!(out_div, b, QR)
@assert allocated_memory_mul == 0
@assert allocated_memory_div == 0

# Testing for correctness:
A = randn(25, 20) 
x = randn(20)
b = randn(25)
out_mul = randn(25)
out_div = randn(20)
true_mul = A * x 
true_div = A \ b 

householder_QR!(A)
QR = A
householder_QR_mul!(out_mul, x, QR)
householder_QR_div!(out_div, b, QR)

# checks whether the results are approximately correct
@assert true_mul ≈ out_mul
@assert true_div ≈ out_div


#----------------------------------------
# Problem e
#----------------------------------------
# YOUR CODE GOES HERE

## I use 'opnorm' to use the operator norm (aka the matrix norm) for a better comparison
using LinearAlgebra: opnorm
## Define the range of matrix sizes to test
sizes = 2 .^ (2:9)
## Define the number of trials per matrix size
trials = 10

# extract the actual Q matrix from the result QR matrix by Householder Reflection
function extract_Q(QR_hh)
    m, n = size(QR_hh)
    Q = Matrix{Float64}(I, m, m)
    for k in n:-1:1
        x = view(QR_hh, k:m, k)
        v = zeros(m-k+1)
        v += x
        v[1] = 1
        beta = 2.0 / (v' * v)
        Q[k:end, k:end] -= beta * v * (v' * Q[k:end, k:end])
    end
    return Q
end

#######################
# Well-conditioned Case
#######################

## Initialize arrays to store the relative errors for each method
# Classical Gram-Schmidt
cgs_qr_errors_well = zeros(length(sizes), trials)
cgs_orth_errors_well = zeros(length(sizes), trials)
# Modified Gram-Schmidt
mgs_qr_errors_well = zeros(length(sizes), trials)
mgs_orth_errors_well = zeros(length(sizes), trials)
# Householder Reflection
hh_qr_errors_well = zeros(length(sizes), trials)
hh_orth_errors_well = zeros(length(sizes), trials)

## Loop over the different matrix sizes and compute the errors for each method
for (i, n) in enumerate(sizes)
    for j in 1:trials
        ## Generate a random matrix of size n-by-n
        local A = randn(n, n)
        
        ## Compute the QR factorization using the classical Gram-Schmidt algorithm
        Qc, Rc = classical_gram_schmidt(A)
        cgs_qr_errors_well[i, j] = opnorm(A - Qc * Rc, Inf) / opnorm(A, Inf)
        cgs_orth_errors_well[i, j] = opnorm(I - Qc' * Qc, Inf)
        
        ## Compute the QR factorization using the modified Gram-Schmidt algorithm
        Qm, Rm = modified_gram_schmidt(A)
        mgs_qr_errors_well[i, j] = opnorm(A - Qm * Rm, Inf) / opnorm(A, Inf)
        mgs_orth_errors_well[i, j] = opnorm(I - Qm' * Qm, Inf)

        ## Compute the QR factorization using the Householder reflection algorithm
        QR_hh = copy(A)
        householder_QR!(QR_hh)
        Qh = extract_Q(QR_hh)
        Rh = triu(QR_hh)  # assuming a square matrix in this experiment
        hh_qr_errors_well[i, j] = opnorm(A - Qh * Rh, Inf) / opnorm(A, Inf)
        hh_orth_errors_well[i, j] = opnorm(I - Qh' * Qh, Inf)
    end
end

# Plot the errors for each method as a function of matrix size
pl_qr_error_well = scatter(sizes, vec(sum(cgs_qr_errors_well, dims=2))./trials, color=(:blue,0.5), label="Classical GS",
    axis=(yscale=log10, xlabel="Matrix Size", ylabel=L"\left\Vert A - QR \right\Vert_{\infty} / \left\Vert A \right\Vert_{\infty}"),
)
scatter!(sizes, vec(sum(mgs_qr_errors_well, dims=2))./trials, marker=:utriangle, color=(:red,0.5), label="Modified GS")
scatter!(sizes, vec(sum(hh_qr_errors_well, dims=2))./trials, marker=:ltriangle, color=(:green,0.5), label="Householder")
axislegend(position=:rb)
save("qr_error_well.pdf", pl_qr_error_well)

pl_orth_error_well = scatter(sizes, vec(sum(cgs_orth_errors_well, dims=2))./trials, color=(:blue,0.5), label="Classical GS",
    axis=(yscale=log10, xlabel="Matrix Size", ylabel=L"\left\Vert I - Q^{\top}Q \right\Vert_{\infty}"),
)
scatter!(sizes, vec(sum(mgs_orth_errors_well, dims=2))./trials, marker=:utriangle, color=(:red,0.5), label="Modified GS")
scatter!(sizes, vec(sum(hh_orth_errors_well, dims=2))./trials, marker=:ltriangle, color=(:green,0.5), label="Householder")
axislegend(position=:rb)
save("orth_error_well.pdf", pl_orth_error_well)

#######################
# Ill-conditioned Case
#######################

function hilbert(n::Int)
    H = zeros(n,n)
    for i in 1:n
        for j in 1:n
            H[i,j] = 1/(i+j-1)
        end
    end
    return H
end

## Initialize arrays to store the relative errors for each method
# Classical Gram-Schmidt
cgs_qr_errors_ill = zeros(length(sizes), trials)
cgs_orth_errors_ill = zeros(length(sizes), trials)
# Modified Gram-Schmidt
mgs_qr_errors_ill = zeros(length(sizes), trials)
mgs_orth_errors_ill = zeros(length(sizes), trials)
# Householder Reflection
hh_qr_errors_ill = zeros(length(sizes), trials)
hh_orth_errors_ill = zeros(length(sizes), trials)

## Loop over the different matrix sizes and compute the errors for each method
for (i, n) in enumerate(sizes)
    for j in 1:trials
        ## Generate the n-th order Hilbert matrix
        local A = hilbert(n)
        
        ## Compute the QR factorization using the classical Gram-Schmidt algorithm
        Qc, Rc = classical_gram_schmidt(A)
        cgs_qr_errors_ill[i, j] = opnorm(A - Qc * Rc, Inf) / opnorm(A, Inf)
        cgs_orth_errors_ill[i, j] = opnorm(I - Qc' * Qc, Inf)
        
        ## Compute the QR factorization using the modified Gram-Schmidt algorithm
        Qm, Rm = modified_gram_schmidt(A)
        mgs_qr_errors_ill[i, j] = opnorm(A - Qm * Rm, Inf) / opnorm(A, Inf)
        mgs_orth_errors_ill[i, j] = opnorm(I - Qm' * Qm, Inf)

        ## Compute the QR factorization using the Householder reflection algorithm
        QR_hh = copy(A)
        householder_QR!(QR_hh)
        Qh = extract_Q(QR_hh)
        Rh = triu(QR_hh)  # assuming a square matrix in this experiment
        hh_qr_errors_ill[i, j] = opnorm(A - Qh * Rh, Inf) / opnorm(A, Inf)
        hh_orth_errors_ill[i, j] = opnorm(I - Qh' * Qh, Inf)
    end
end

# Plot the errors for each method as a function of matrix size
pl_qr_error_ill = scatter(sizes, vec(sum(cgs_qr_errors_ill, dims=2))./trials, color=(:blue,0.5), label="Classical GS",
    axis=(yscale=log10, xlabel="Matrix Size", ylabel=L"\left\Vert A - QR \right\Vert_{\infty} / \left\Vert A \right\Vert_{\infty}"),
)
scatter!(sizes, vec(sum(mgs_qr_errors_ill, dims=2))./trials, marker=:utriangle, color=(:red,0.5), label="Modified GS")
scatter!(sizes, vec(sum(hh_qr_errors_ill, dims=2))./trials, marker=:ltriangle, color=(:green,0.5), label="Householder")
axislegend(position=:rb)
save("qr_error_ill.pdf", pl_qr_error_ill)

pl_orth_error_ill = scatter(sizes, vec(sum(cgs_orth_errors_ill, dims=2))./trials, color=(:blue,0.5), label="Classical GS",
    axis=(yscale=log10, xlabel="Matrix Size", ylabel=L"\left\Vert I - Q^{\top}Q \right\Vert_{\infty}"),
)
scatter!(sizes, vec(sum(mgs_orth_errors_ill, dims=2))./trials, marker=:utriangle, color=(:red,0.5), label="Modified GS")
scatter!(sizes, vec(sum(hh_orth_errors_ill, dims=2))./trials, marker=:ltriangle, color=(:green,0.5), label="Householder")
axislegend(position=:rb)
save("orth_error_ill.pdf", pl_orth_error_ill)
