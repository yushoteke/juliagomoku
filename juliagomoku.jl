using LinearAlgebra

mutable struct gomoku
    B::Array{Bool,3}
    L::Int64
    player_playing::Int64
    is_terminal::Bool
    n_in_row::Int64
end

gomoku(m,n_in_row)=gomoku(zeros(Bool,m,m,2),m,1,false,n_in_row)
valid_moves(G::gomoku)=findall(x->x==0,G.B[:,:,1]+G.B[:,:,2])
current_perspective(G::gomoku)=G.player_playing==1 ? G.B : G.B[:,:,2:-1:1]

function print_board(G::gomoku)
    println("=========================")
    tmp=G.B[:,:,1]+2G.B[:,:,2]
    for i=1:G.L
        println(tmp[i,:])
    end
    println("=========================")
end

function play_move!(G::gomoku,m::CartesianIndex{2})
    if G.is_terminal==false
        if sum(G.B[m,:])>0
            println("invalid move, stone already exist")
        elseif m[1]>G.L || m[1]<1 || m[2]>G.L || m[2]<1
            println("invalid move. out of bounds")
        else
            G.B[m,G.player_playing]=true
            G.is_terminal=terminal_check(G,m)
            G.player_playing=3-G.player_playing
        end
    else
        println("game already ended")
    end
end

function Nconsecutive(A::Array{Bool,1},n::Int64)
    cnt = 0
    for i=1:length(A)
        cnt = A[i] * (cnt + A[i])
        if cnt >= n
            return true
        end
    end
    return false
end

function terminal_check(G::gomoku,m::CartesianIndex{2})
    #given last move at m, check if game ended
    if G.B[m,1]==true
        subboard=G.B[:,:,1]
    elseif G.B[m,2]==true
        subboard=G.B[:,:,2]
    else
        return false
    end

    horizontal=subboard[m[1],:]
    vertical  =subboard[:,m[2]]
    ldiagonal =diag(subboard,m[2]-m[1])
    rdiagonal = diag(subboard[:,G.L:-1:1],G.L+1-m[2]-m[1])

    return Nconsecutive(horizontal,G.n_in_row) ||
            Nconsecutive(vertical,G.n_in_row) ||
            Nconsecutive(ldiagonal,G.n_in_row) ||
            Nconsecutive(rdiagonal,G.n_in_row)
end
