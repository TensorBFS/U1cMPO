"""
    pauli(M)

Generate Pauli matrix (σ0, σ+, and σ-).
"""
function pauli(M::Integer)
    if M == 0
        return init_diag_u1m([1, 1], [1., -1.])
    elseif M == 1
        return init_u1m_by_params([1, 1], 1, [1.0])
    elseif M == -1
        return init_u1m_by_params([1, 1], -1, [1.0])
    end
    throw(error("input can only be 0 or ±1"))
end

"""
    pauli_id()

Generate 2×2 identity matrix.
"""
function pauli_id()
    eye_u1m(Float64, [1, 1])
end
