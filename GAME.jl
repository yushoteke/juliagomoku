include("MonteCarloTreeSearch.jl")
include("gomokunet.jl")
using StatsBase

function play_gomoku(p1,p2;n=5,n_in_row=3,n_games=1)
    G=gomoku(n,n_in_row)
    print_board(G)

    data=[]
    while !G.is_terminal
        NN = G.player_playing==1 ? p1 : p2
        stronger_pi = get_stronger_pi(G,NN)
        next_move = StatsBase.sample(CartesianIndices((1:G.L,1:G.L)),pweights(stronger_pi))
        push!(data,(G.B,stronger_pi,0.0))
        play_move!(G,next_move)
        print_board(G)
    end
    data[end,3]=
    return data
end

play_gomoku(NN,NN)
