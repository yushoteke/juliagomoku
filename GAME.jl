include("MonteCarloTreeSearch.jl")
include("gomokunet.jl")
using StatsBase

function play_gomoku(p1,p2;n=5,n_in_row=3,debug=true)
    G=gomoku(n,n_in_row)
    if debug
        print_board(G)
    end

    p1_data=[]
    p2_data=[]
    while !G.is_terminal
        NN = G.player_playing==1 ? p1 : p2
        stronger_pi = get_stronger_pi(G,NN)
        next_move = StatsBase.sample(CartesianIndices((1:G.L,1:G.L)),pweights(stronger_pi))
        push!(G.player_playing==1 ? p1_data : p2_data,[copy(current_perspective(G)),stronger_pi,0.0])
        play_move!(G,next_move)
        if debug
            print_board(G)
        end
    end
    for i in p1_data
        i[3]=G.player_playing==1 ? -1.0 : 1.0
    end
    for i in p2_data
        i[3]=G.player_playing==2 ? -1.0 : 1.0
    end
    return p1_data,p2_data,3-G.player_playing
end

#play_gomoku(NN1,NN2)

function gather_data(p1,p2;n_games=40,debug=true)
    p1_data=[]
    p2_data=[]
    wins=[0,0]
    for i=1:n_games
        if i%5==0
            println("playing game ",i)
        end
        if rand() > 0.5
            tmp1,tmp2,winner=play_gomoku(p1,p2,debug=debug)
        else
            tmp2,tmp1,winner=play_gomoku(p2,p1,debug=debug)
        end
        p1_data=cat(p1_data,deepcopy(tmp1),dims=1)
        p2_data=cat(p2_data,deepcopy(tmp2),dims=1)
        wins[winner]+=1
    end
    println("p1 won ",wins[1]," games")
    println("p2 won ",wins[2]," games")
    return p1_data,p2_data,wins[1]/(wins[2]+wins[1])
end

my_crossentropy(x,y)=x.*log.(y)+(1 .-x).*log.(1 .-y) |> mean

function train_player(data,NN::player;batchsize=32,eta=0.0001,epoch=10)
    w,h,c=size(data[1][1])
    input=zeros(Float32,w,h,c,batchsize)
    better_pi=zeros(w,h,1,batchsize)
    real_v=zeros(1,1,1,batchsize)

    function player_loss(x,better_pi,real_v)
        pi,v=NN(x)
        return 0.001sum(norm,params(NN))+Flux.mse(v,real_v)-my_crossentropy(better_pi,pi)
    end
    opt = ADAM(eta)

    println("training...")
    for i=1:epoch
        for i=1:batchsize
            tmp=rand(data)
            input[:,:,:,i]=tmp[1]
            better_pi[:,:,:,i]=tmp[2]
            real_v[1,1,1,i]=tmp[3]
        end
        Flux.train!(player_loss,params(NN),[(input,better_pi,real_v)],opt)
    end
end

function self_play(p1;iterations=5,n_games=50,debug=false)
    p2=deepcopy(p1)
    for iter=1:iterations
        p1_data,p2_data,win_rate=gather_data(p1,p2,n_games=n_games,debug=debug)
        println("iteration ",iter," winrate ",win_rate)
        train_player(p1_data,p1)
        if win_rate>0.7
            p2=deepcopy(p1)
            println("best player changed")
        end
    end
    return p1
end
p1=player(5)
better_player=self_play(p1,iterations=1,n_games=1,debug=true)
#compile code

p1=player(5)
better_player=self_play(p1,iterations=30,n_games=50)

batchsize=32
w,h,c=size(p1_data[1][1])
input=zeros(Float32,w,h,c,batchsize)
better_pi=zeros(Float32,w,h,1,batchsize)
real_v=zeros(1,1,1,batchsize)
for i=1:batchsize
    tmp=rand(p1_data)
    input[:,:,:,i]=tmp[1]
    better_pi[:,:,:,i]=tmp[2]
    real_v[1,1,1,i]=tmp[3]
end



function player_loss(x,better_pi,real_v)
    pi,v=p1(x)
    return 0.001sum(norm,params(p1))+Flux.mse(v,real_v)-my_crossentropy(better_pi,pi)
end
player_loss(input,better_pi,real_v)
opt = ADAM(0.0001)

Flux.train!(player_loss,params(p1),[(input,better_pi,real_v)],opt)
