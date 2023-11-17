#----------------------------------------
# a: reduce a symmetric matrix A to Hessenberg form using Householder reflections
#    operate in place, overwriting the input matrix with Hessenberg form
#----------------------------------------
# This function takes in a matrix A and returns 
# a reduced QR factorization with factors Q and R.
# It should not modify A
function classical_gram_schmidt(A)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)

    for j in 1:n
        q_hat_j = copy(A[:, j])
        for i in 1:j-1
            R[i, j] = Q[:, i]' * A[:, j]
            q_hat_j -= R[i, j] * Q[:, i]
        end
        R[j, j] = norm(q_hat_j)
        Q[:, j] = q_hat_j / R[j, j]
    end
    
    return Q, R
end

#----------------------------------------
# b: run a single iteration of the unshifted QR algorithm 
#    using Givens rotations to implement QR factorization on T_k
#    input = T_k hessenberg from 
#    output = T_{k+1} Hessenberg form
#----------------------------------------
# This function takes in a matrix A and returns 
# a reduced QR factorization with factors Q and R.
# It should not modify A
function modified_gram_schmidt(A)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)

    for j in 1:n
        q_hat_j = copy(A[:, j])
        for i in 1:j-1
            R[i, j] = Q[:, i]' * q_hat_j
            q_hat_j -= R[i, j] * Q[:, i]
        end
        R[j, j] = norm(q_hat_j)
        Q[:, j] = q_hat_j / R[j, j]
    end
    
    return Q, R
end

#----------------------------------------
# c: ren the practical QR iteration with both the Single-Shift and Wilkinson Shift. 
#    using QR iteration in b, with criteria for when to implement deflation and when to terminate QR iteration    
#----------------------------------------
# This function takes in a matrix A 
# and computes its QR factorization in place,
# using householder reflections.
# It should not allocate any memory.  
function householder_QR!(A)
    m, n = size(A)
    for k in 1:n
        x = view(A, k:m, k)
        gamma = sign(x[1]) * norm(x)
        denom = (x[1] + gamma)
        x[1] += gamma
        x ./= denom
        
        beta = 2 / (norm(x) ^ 2)
        for j in k+1:n
            inner = x' * view(A, k:m, j)
            for i in k:m
                A[i, j] = A[i, j] - beta * x[i-k+1] * inner
            end
        end
        
        for i in k+1:m
            A[i, k] = x[i-k+1]
        end
        
        A[k, k] = -gamma
    end
end

#----------------------------------------
# d: design an experiment that evaluates your practical QR algorithm with shifts
#    include a semi-log plot showing the rate of convergence of Single-Shift and Wilkinson Shift to compare
#----------------------------------------
# These two functions take in the housholder
# QR factorization from part c and multiply them
# to a vector (mul) or solve the least squares 
# problem in A (div), in place.
# They should not allocate any memory and instead
# use the preallocated output vector to record the result. 
function householder_QR_mul!(out, x, QR)
    m, n = size(QR)
    @assert size(out,1) == m && length(x) == n

    for i in 1:m
        out[i] = 0
    end

    # Rx
    for j in 1:n
        for i in 1:j
            out[i] += QR[i, j] * x[j]
        end
    end

    # Q(Rx)
    for k in n:-1:1
        vk = view(QR, k:m, k)
        prev_value = vk[1]
        vk[1] = 1
        beta = 2.0 / (vk' * vk)
        
        # memory allocation version
        # out[k:m] -= beta * vk * (vk' * out[k:m])
        dot_vk_y = vk' * view(out, k:m)
        for j in k:m
            out[j] -= beta * dot_vk_y .* vk[j-k+1]
        end
        vk[1] = prev_value
    end
end

function householder_QR_div!(out, b, QR)
    # YOUR CODE HERE
    m, n = size(QR)
    @assert size(out,1) == n && length(b) == m

    # Q^H b
    for k in 1:n
        vk = view(QR, k:m, k)
        prev_value = vk[1]
        vk[1] = 1
        beta = 2.0 / (vk' * vk)
        # memory allocation version
        # b[k:m] -= beta * vk * (vk' * b[k:m])
        dot_vk_b = vk' * view(b, k:m)
        for j in k:m
            b[j] -= beta * dot_vk_b .* vk[j-k+1]
        end
        vk[1] = prev_value
    end
    
    # back substitution
    # Rx = Q^H b
    # x = R^-1 Q^H b
    for j in n:-1:1
        b[j] /= QR[j, j]
        for i in 1:(j-1)
            b[i] -= QR[i, j] * b[j]
        end
    end
    
    # update output (x) 
    for i in 1:n
        out[i] = b[i]
    end
end
