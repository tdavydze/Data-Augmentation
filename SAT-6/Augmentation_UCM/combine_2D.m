function A = combine_2D(A, B, M)
A = A .* (M == 0) + B .* (M == 1);