#==========================================================
Gates for pulse optimizer
==========================================================#

using LinearAlgebra
using Printf
import Base.*, Base.kron

"""
A Gate only be manually initialized as a no-guard-state matrix, then extended later using extend().
"""
mutable struct Gate
    mat::Matrix{ComplexF64}     # full matrix
    rawmat::Matrix{ComplexF64}  # matrix without guard states
    Ne::Vector{Int64}           # num. essential levels per bit
    Nh::Vector{Int64}           # hidden levels per bit (essential levels that are not initial conditions)
    Ng::Vector{Int64}           # guard levels per bit
    Nt::Vector{Int64}           # convenience, Nt = Ne+Ng+Nh
    Nobj::Int64                 # convenience, Nobj = number of columns of mat
    Nout::Int64                 # convenience, Nout = number of rows of mat
    nbits::Int64                # convenience, nbits = length(Ne)
    name::String                # name of the gate (for printing and auto file naming)

    # can be constructed in many ways
    Gate(mat) =                             new(mat, copy(mat), [size(mat,2)], [0], [size(mat,1)-size(mat,2)], [size(mat,1)], size(mat,2), size(mat,1), 1, "unnamed")
    Gate(name, mat) =                       new(mat, copy(mat), [size(mat,2)], [0], [size(mat,1)-size(mat,2)], [size(mat,1)], size(mat,2), size(mat,1), 1, name)
    Gate(name, mat, Ne) =                   new(mat, copy(mat), Ne, zeros(length(Ne)), zeros(length(Ne)), Ne, size(mat,2), size(mat,1), length(Ne), name)
    Gate(name, mat, Ne, Nh) =               new(mat, copy(mat), Ne, Nh, zeros(length(Ne)), Ne+Nh, size(mat,2), size(mat,1), length(Ne), name)
    
    # these are generally reserved for internal use (i.e. in extend())
    Gate(name, mat, rawmat, Ne, Ng) =       new(mat, rawmat, Ne, zeros(length(Ne)), Ng, Ne+Ng+Nh, size(mat,2), size(mat,1), length(Ne), name)
    Gate(name, mat, rawmat, Ne, Nh, Ng) =   new(mat, rawmat, Ne, Nh, Ng, Ne+Ng+Nh, size(mat,2), size(mat,1), length(Ne), name)
end

*(a::Gate, b::Gate) = multiply(a, b)
*(a::Gate, v::Vector) = a.mat * v
Base.show(io::IO, ::MIME"text/plain", a::Gate) = (println(a.name); display(a.mat); println("Ne : $(a.Ne)\nNh : $(a.Nh)\nNg : $(a.Ng)"))
Base.println(io::IO, a::Gate) = (println(a.name); display(a.mat); println("Ne : $(a.Ne)\nNh : $(a.Nh)\nNg : $(a.Ng)"))

"""
Compose two gates. Works with guard levels.
TODO: how should this work with hidden levels?
"""
function multiply(a::Gate, b::Gate)
    @assert a.Ne == b.Ne
    raw = Gate("$(a.name) * $(b.name)", a.rawmat*b.rawmat, a.Ne)
    return extend(raw, [max(ag, bg) for (ag,bg) in zip(a.Ng, b.Ng)])
end

"""
Kronecker product.
"""
function kron(a, b, increase_bits=true)
    if increase_bits
        Ne = vcat(b.Ne, a.Ne)
        Nh = vcat(b.Nh, a.Nh)
        Ng = vcat(b.Ng, a.Ng)
    else
        @assert a.nbits == b.nbits
        @assert a.Ne == b.Ne
        @assert sum(a.Nh + b.Nh) == 0
        Ne = a.Ne + b.Ne
        Nh = zeros(length(a.Ne))
        Ng = [max(ag, bg) for (ag,bg) in zip(a.Ng, b.Ng)]
    end
    return extend(Gate("$(a.name) ⊗ $(b.name)", kron(a.rawmat, b.rawmat), Ne, Nh), Ng)
end


"""
Get bit pattern from numerical value (i.e. new gate row).
`nstates` is a vector of states for each bit.
ex:
bits_from_idx(0, [2,2]) = [0,0]
bits_from_idx(2, [2,2]) = [1,0]
bits_from_idx(4, [4,2]) = [1,0]

NOTE: `nstates` uses the reversed ordering:
if qubit A has 2 states and qutrit B has 3 states, with bit vector |ab> = |12> = |5>,
the `idx` argument would be 5 and the `nstates` argument would be [3,2], returning [1,2].
This is to be consistent with Juqbox's ordering of Ne and Ng.
"""
function bits_from_idx(idx, nstates)
    nbits = length(nstates)
    indices = zeros(Int, nbits)
    for bit in 1:nbits
        bit_value = prod(nstates[1:nbits-bit])
        curbitval = div(idx, bit_value)
        indices[bit] = curbitval
        idx -= bit_value * curbitval
    end
    return indices
end

"""
Get numerical value (i.e. gate row index) from bit pattern
idx_from_bits([0,0], [2,2]) = 0
idx_from_bits([1,0], [2,2]) = 2
idx_from_bits([1,0], [4,2]) = 4

NOTE: `nstates` uses the reversed ordering:
if qubit A has 2 states and qutrit B has 3 states, with bit vector |ab> = |12> = |5>,
the `bits` argument would be [1,2] and the `nstates` argument would be [3,2], returning 5.
This is to be consistent with Juqbox's ordering of Ne and Ng.
"""
function idx_from_bits(bits, nstates)
    nbits = length(nstates)
    idx = 0
    for bit in 1:nbits
        bit_value = prod(nstates[1:nbits-bit])
        idx += bits[bit]*bit_value
    end
    return Int(idx)
end

"""
Extend a multi-bit matrix (add guard levels)
Examples:
`extend(Xp(3), 2)` adds 2 guard states to the qutrit
`extend(CNOT, [3,3])` adds 3 guard states to each qubit
"""
function extend(gate, Ng)
    if maximum(gate.Ng) > 0
        println("WARNING: extend is removing previous guard states and replacing them with $Ng")
    end
    
    if typeof(Ng) === Int64
        Ng = [Ng]
    end
    @assert length(Ng) == gate.nbits

    if length(Ng) == 1
        # just append zeros to the bottom of the array
        extradims = zeros(Ng[1], gate.Nobj)
        return Gate(gate.name, vcat(gate.rawmat, extradims), gate.rawmat, gate.Ne, gate.Nh, Ng)
    else
        # need to fill in zeros at different spots in the array
        ntot = Ng+gate.Ne+gate.Nh
        Nout = prod(ntot)
        newgate = zeros(ComplexF64, Nout, gate.Nobj)

        oldrow = 1
        for r in 1:Nout
            bits = bits_from_idx(r-1, ntot)
            # if all bits are essential
            if prod(reverse(bits) .< gate.Ne+gate.Nh)
                newgate[r,:] = gate.rawmat[oldrow,:]
                oldrow += 1
            end
        end
        
        return Gate(gate.name, newgate, gate.rawmat, gate.Ne, gate.Nh, Ng)
    end
end

i1 = Gate("i1", Matrix(1I,1,1))
i2 = Gate("i2", Matrix(1I,2,2))
i3 = Gate("i3", Matrix(1I,3,3))
i4 = Gate("i4", Matrix(1I,4,4))

X = Gate("X", Matrix{ComplexF64}(
    [0 1;
     1 0]
))

Y = Gate("Y", Matrix{ComplexF64}(
    [0  -im;
     im   0]
))

Z = Gate("Z", Matrix{ComplexF64}(
    [1  0;
     0 -1;]
))

h = 1/sqrt(2)

H = Gate("H", Matrix{ComplexF64}(
    [h  h; 
     h -h]
))

"generalized Hadamard (quantum Fourier transform) gate"
function Had(nstates)
    H = zeros(ComplexF64,nstates,nstates)
    for row in 1:nstates
        for col in 1:nstates
            H[row,col] = exp(1im*2*pi*(row-1)*(col-1)/nstates)/sqrt(nstates)
        end
    end
    Gate("H_$nstates", H)
end

"generalized Z gate"
function Zn(nstates)
    Z = zeros(ComplexF64,nstates,nstates)
    for idx in 1:nstates
        Z[idx,idx] = exp(im*2*pi*(idx-1)/nstates)
    end
    Gate("Z_$nstates", Z)
end

"generalized T gate"
function Tn(nstates)
    T = zeros(ComplexF64,nstates,nstates)
    for idx in 1:nstates
        T[idx,idx] = exp(im*2*pi*(idx-1)/(4*nstates))
    end
    Gate("T_$nstates", T)
end

"X+1 gate" 
function Xp(nstates)
    Xp = zeros(ComplexF64,nstates,nstates)
    for idx in 1:nstates-1
        Xp[idx+1,idx] = 1.0
    end
    Xp[1,nstates] = 1.0
    Gate("X+_$nstates", Xp)
end

"X-1 gate"
function Xm(nstates)
    Xm = zeros(ComplexF64,nstates,nstates)
    for idx in 1:nstates-1
        Xm[idx,idx+1] = 1.0
    end
    Xm[nstates,1] = 1.0
    Gate("X-_$nstates", Xm)
end

function swap_0_d(nstates)
    gate = Matrix{ComplexF64}(1I,nstates,nstates)
    gate[1,1] = 0.0
    gate[1,nstates] = 1.0
    gate[nstates,1] = 1.0
    gate[nstates,nstates] = 0.0
    return Gate("dswap_$nstates", gate, [nstates])
end


function SWAP(nstates)
    swap = zeros(ComplexF64, nstates^2, nstates^2)
    for s1 in 0:nstates-1
        for s2 in 0:nstates-1
            idxin = s1*nstates + s2 + 1
            idxout = s2*nstates + s1 + 1
            swap[idxout, idxin] = 1.0
        end
    end
    Gate("SWAP_$nstates", swap, [nstates, nstates])
end

"""Make a gate matrix that applies `gate` on `states`
with total dimensions `(nstates, nstates)`

Ex: X_12 with 2 guard states = gate_on_states([1,2],X,3,2)"""
function gate_on_states(states, gate, nstates, nguard=0)
    @assert gate.Nobj == gate.Nout
    newgate = Matrix{ComplexF64}(1I,nstates,nstates)
    for i in 1:length(states)
        s = states[i]
        newgate[s+1,s+1] = 0.0
        for j in 1:length(states)
            s1 = states[j]
            newgate[s+1,s1+1] = gate.mat[i,j]
        end
    end
    newgate = Gate(gate.name * "_" * prod([string(s) for s in states]), newgate)
    extend(newgate, nguard)
end

"""Generalized c gate for arbitrary num. of bits.
NOTE: uses Juqbox ordering (see examples below)
TODO: support a gate over multiple target bits
params:
`control_bits`: control bit(s)
`target_bit`: target bit
`Ne`: array of num essential states per bit
`Ng`: array of num guard states per bit
`control_states`: state(s) of control_bits to apply target gate on. if `control_bits` is
an array, this is a 2D array.
`gate`: gate to apply to target.

Ex: standard CNOT gate, 3 guard states on each bit:
`make_C_gate(1,0,[2,2],[3,3],1,X)

Ex: Toffoli gate, no guard levels:
`make_C_gate([1,2],0,[2,2,2],[0,0,0],[1,1],X)`

Ex: 1 qubit and 2 qutrits, controlled X+ gate on bit 2 when bit 0 is in |1> state AND bit 1 is either in |1> or |2> states:
`make_C_gate([0,1],2,[2,3,3],[2,2,2],[[1],[1,2]], Xp(3))`
"""
function make_C_gate(control_bits, target_bit, Ne, Ng, control_states, gate, name=nothing)
    # make ctrl bit an array
    if isa(control_bits, Number)
        control_bits = [control_bits]
    end
    # make ctl states a 2d array
    if isa(control_states, Number)
        statestring = string(control_states)
        control_states = [[control_states]]
    elseif isa(control_states[1], Number)
        statestring = prod([string(x) for x in control_states[1]])
        if length(control_bits) == 1
            control_states = [control_states]
        else
            control_states = [[c] for c in control_states]
        end
    else
        statestring = prod([string(x) for x in control_states])
    end
    essential_size = prod(Ne)

    newgate = zeros(ComplexF64, essential_size, essential_size)

    for c in 0:essential_size-1
        bits_in = reverse(bits_from_idx(c, Ne)) #reverse to get to Juqbox ordering
        controls_on = true
        # all control bits must be in proper state for gate to be applied
        for j in 1:length(control_bits)
            if !(bits_in[control_bits[j]+1] in control_states[j])
                controls_on = false
            end
        end
        for r in 0:essential_size-1
            bits_out = reverse(bits_from_idx(r, Ne))

            controls_unchanged = true
            for i in 1:length(bits_in)
                if !(i-1 == target_bit || bits_in[i] == bits_out[i])
                    controls_unchanged = false
                end
            end

            if controls_on && controls_unchanged
                state_in = bits_in[target_bit+1]
                state_out = bits_out[target_bit+1]
                newgate[r+1, c+1] = gate.rawmat[state_out+1, state_in+1]
            elseif r == c
                newgate[r+1, c+1] = 1.0
            end
        end
    end

    if name === nothing
        name = "C_"*statestring*"_$(gate.name)"
    end

    no_guard = Gate(name, newgate, Ne)
    return extend(no_guard, Ng)
end

"""swap qudit order in a 2-bit gate"""
function swap_bits(gate)
    newmatraw = zeros(ComplexF64, size(gate.rawmat))
    for c in 0:gate.Nobj-1
        for r in 0:(prod(gate.Ne+gate.Nh))-1
            newc = 1+idx_from_bits(reverse(bits_from_idx(c, gate.Ne)), reverse(gate.Ne))
            newr = 1+idx_from_bits(reverse(bits_from_idx(r, gate.Ne+gate.Nh)), reverse(gate.Ne+gate.Nh))
            newmatraw[newr, newc] = gate.rawmat[r+1, c+1]
        end
    end

    return extend(Gate("$(gate.name)_swapped", newmatraw, reverse(gate.Ne), reverse(gate.Nh)), reverse(gate.Ng))
end

function encval(s)
    if s == 0
        return "00"
    elseif s == 1
        return "01"
    elseif s == 2
        return "10"
    elseif s == 3
        return "11"
    else
        return nothing
    end
end

"""
Create a length-n vector [0,...,0,1,0,...,0] with index `state` equal to 1, and the rest 0.
"""
function onevec(state, n)
    v = zeros(n)
    v[state+1] = 1
    v
end

"""
Print the state transitions of a single-qudit gate.

`enc`: if true, treat 
"""
function transitions(gate, enc=false)
    println("Nt: $(gate.Nt)")
    if gate.nbits == 1
        if enc && gate.Ne[1] != 4
            println("Warning: Ne != 4, cannot treat as encoded")
        end

        for s in 0:gate.Ne[1]-1
            v = onevec(s, gate.Ne[1])
            out_vec = gate * v
            out_state = indexin(1, out_vec)[1]
            if out_state === nothing
                println("WARNING: only for use with unitaries consisting of 1s and 0s")
                return
            end
            out_state -= 1
            if enc
                @printf("%s => %s\n", encval(s), encval(out_state))
            else
                @printf("%d => %d\n", s, out_state)
            end
        end
    elseif gate.nbits == 2
        enc1_in = enc && (gate.Ne)[2] == 4
        enc2_in = enc && (gate.Ne)[1] == 4
        enc1_out = enc && (gate.Ne+gate.Nh)[2] == 4
        enc2_out = enc && (gate.Ne+gate.Nh)[1] == 4

        for s1 in 0:gate.Ne[2]-1
            for s2 in 0:gate.Ne[1]-1
                in_vec = onevec(idx_from_bits([s1,s2], gate.Ne), gate.Nobj)
                out_vec = gate.mat * in_vec
                out_state = indexin(1, out_vec)[1]
                if out_state === nothing
                    println("WARNING: only for use with unitaries consisting of 1s and 0s")
                    return
                end
                out_state -= 1
                ss = bits_from_idx(out_state, gate.Nt)
                s1new = ss[1]
                s2new = ss[2]
                if enc1_in || enc2_in || enc1_out || enc2_out
                    @printf("%s %s => %s %s\n", enc1_in ? encval(s1) : s1, enc2_in ? encval(s2) : s2, enc1_out ? encval(s1new) : s1new, enc2_out ? encval(s2new) : s2new)
                else
                    @printf("%d%d => %d%d\n", s1, s2, s1new, s2new)
                end
            end
        end
    end
end

"""
Convert vector to matrix e.g. [1, 2, 3, 4] => [1 2; 3 4].
Native Julia function `reshape` would produce [1 3; 2 4] instead.
""" 
function from_vec(m, Ne, name)
    nstates = Int(sqrt(length(m)))
    m = hcat(m)
    g = zeros(ComplexF64, (nstates, nstates))
    for i in 1:nstates
        g[i,:] = m[1+nstates*(i-1):nstates*i]
    end
    return Gate(name, g, Ne)
end

"""
Vector of each possible ntuple of bits
e.g. Nt = [3,2] =>
[(0,0), (0,1), (0,2),
 (1,0), (1,1), (1,2)]
"""
function all_bits(Nt)
    return vec(collect(product([0:n-1 for n in reverse(Nt)]...)))
end

# Gates for "logical encoding" of 2 bits into one quart

internal_CNOT_c0 = extend(Gate("internal_CNOT_c0", Matrix{ComplexF64}(
    [1 0 0 0;
     0 1 0 0;
     0 0 0 1;
     0 0 1 0]
)), 1)

internal_CNOT_c1 = extend(Gate("internal_CNOT_c1", Matrix{ComplexF64}(
    [1 0 0 0;
     0 0 0 1;
     0 0 1 0;
     0 1 0 0]
)), 1)

internal_SWAP = extend(Gate("internal_SWAP", Matrix{ComplexF64}(
    [1 0 0 0;
     0 0 1 0;
     0 1 0 0;
     0 0 0 1]
)), 1)

internal_CPHASE = extend(Gate("internal_CPHASE", Matrix{ComplexF64}(
    [1 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 -1]
)), 1)

CPHASE = make_C_gate(1,0,[2,2],[3,3],[1],Z)
CNOT = make_C_gate(1,0,[2,2],[2,2],[1],X)
CNOT.name = "CNOT"

SWAP2 = extend(SWAP(2),[2,2])
SWAP3 = extend(SWAP(3),[2,2])
SWAP4 = extend(SWAP(4),[1,1])

# gates on ququart-encoded qubits
CNOT_enc_01 = gate_on_states([2,3],X,4,2)
CNOT_enc_10 = gate_on_states([1,3],X,4,2)

CNOT_enc_00 = make_C_gate(1,0,[4,4],[1,1],[2,3],kron(X,i2), "CNOT_enc_00") # cnot_low_to_low
CNOT_enc_01 = make_C_gate(1,0,[4,4],[1,1],[2,3],kron(i2,X), "CNOT_enc_01")
CNOT_enc_10 = make_C_gate(1,0,[4,4],[1,1],[1,3],kron(X,i2), "CNOT_enc_10")
CNOT_enc_11 = make_C_gate(1,0,[4,4],[1,1],[1,3],kron(i2,X), "CNOT_enc_11")
CPHASE_2enc_01 = make_C_gate(1,0,[4,4],[1,1],[2,3],kron(i2,Z))

# "partial" encoded gates
CNOT_q_c1 = make_C_gate(1,0,[2,4],[2,1],[1,3],X, "CNOT_q_c1")
CNOT_q_c0 = make_C_gate(1,0,[2,4],[2,1],[2,3],X, "CNOT_q_c0")
CNOT_q_t0 = make_C_gate(1,0,[4,2],[1,2],[1],kron(X, i2), "CNOT_q_t0")
CNOT_q_t1 = make_C_gate(1,0,[4,2],[1,2],[1],kron(i2, X), "CNOT_q_t1")

SWAP_q_0 = CNOT_q_c0*swap_bits(CNOT_q_t0)*CNOT_q_c0 # swap_q_first
SWAP_q_0.name = "SWAP_q_0"
SWAP_q_1 = CNOT_q_c1*swap_bits(CNOT_q_t1)*CNOT_q_c1
SWAP_q_1.name = "SWAP_q_1"

# encoded-encoded qubit swaps
SWAP_enc_00 = CNOT_enc_00 * swap_bits(CNOT_enc_00) * CNOT_enc_00
SWAP_enc_01 = CNOT_enc_01 * swap_bits(CNOT_enc_10) * CNOT_enc_01
SWAP_enc_10 = CNOT_enc_10 * swap_bits(CNOT_enc_01) * CNOT_enc_10
SWAP_enc_11 = CNOT_enc_11 * swap_bits(CNOT_enc_11) * CNOT_enc_11
SWAP_enc_00.name = "SWAP_enc_00"
SWAP_enc_01.name = "SWAP_enc_01"
SWAP_enc_10.name = "SWAP_enc_10"
SWAP_enc_11.name = "SWAP_enc_11"

X_enc_0 = extend(kron(X, i2, false), 2)
X_enc_1 = extend(kron(i2, X, false), 2)
X_enc_both = extend(kron(X, X, false), 2)
X_enc_0.name = "X_enc_0"
X_enc_1.name = "X_enc_1"
X_enc_both.name = "X_enc_both"

# z,gamma,e = 1,1,1
T_3 = [
    1.0 0.0 0.0 ;
    0.0 0.1736481777+0.984807753*im 0.0 ;
    0.0 0.0 -0.9396926208+0.3420201433*im 
]

# z,gamma,e = 1,1,1
T_5 = [
    1.0 0.0 0.0 0.0 0.0 ;
    0.0 0.3090169944+-0.9510565163*im 0.0 0.0 0.0 ;
    0.0 0.0 -0.8090169944+0.5877852523*im 0.0 0.0 ;
    0.0 0.0 0.0 1.0 0.0 ;
    0.0 0.0 0.0 0.0 0.3090169944+-0.9510565163*im 
]

# z,gamma,e = 1,1,1
T_7 = [
    1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
    0.0 -0.222520934+-0.9749279122*im 0.0 0.0 0.0 0.0 0.0 ;
    0.0 0.0 0.6234898019+0.7818314825*im 0.0 0.0 0.0 0.0 ;
    0.0 0.0 0.0 -0.9009688679+0.4338837391*im 0.0 0.0 0.0 ;
    0.0 0.0 0.0 0.0 -0.222520934+-0.9749279122*im 0.0 0.0 ;
    0.0 0.0 0.0 0.0 0.0 0.6234898019+0.7818314825*im 0.0 ;
    0.0 0.0 0.0 0.0 0.0 0.0 0.6234898019+-0.7818314825*im 
]

encmat = [
    1 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0;
    0 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0;
    0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 1 0;
    0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 1
]
enc = extend(Gate("encode", encmat, [2,4]), [2,1])
enc_hidden = extend(Gate("encode", encmat[:,1:4], [2,2], [0,2]),[2,1])