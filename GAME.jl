include("MonteCarloTreeSearch.jl")
include("gomokunet.jl")
using StatsBase

function play_gomoku(p1,p2;n=5,n_in_row=3,n_games=1)
    G=gomoku(n,n_in_row)
    print_board(G)

    p1_data=[]
    p2_data=[]
    while !G.is_terminal
        NN = G.player_playing==1 ? p1 : p2
        stronger_pi = get_stronger_pi(G,NN)
        next_move = StatsBase.sample(CartesianIndices((1:G.L,1:G.L)),pweights(stronger_pi))
        push!(G.player_playing==1 ? p1_data : p2_data,[G.B,stronger_pi,0.0])
        play_move!(G,next_move)
        print_board(G)
    end
    for i in p1_data
        i[3]=G.player_playing==1 ? -1.0 : 1.0
    end
    for i in p2_data
        i[3]=G.player_playing==2 ? -1.0 : 1.0
    end
    return p1_data,p2_data
end

play_gomoku(NN,NN)

function gather_data(p1,p2;n_games=40)
    p1_data=[]
    p2_data=[]
    for i=1:n_games
        if rand() > 0.5
            tmp1,tmp2=play_gomoku(p1,p2)
        else
            tmp2,tmp1=play_gomoku(p2,p1)
        end
        p1_data=cat(p1_data,deepcopy(tmp1),dims=1)
        p2_data=cat(p2_data,deepcopy(tmp2),dims=1)
    end
    return p1_data,p2_data
end

@time gather_data(NN1,NN2,n_games=3)
