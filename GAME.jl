include("MonteCarloTreeSearch.jl")
include("gomokunet.jl")
using StatsBase
using CuArrays
using BSON:@save
using Dates

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
    for i=1:n_games
        if i%10==0
            println("playing game ",i)
        end
        if rand() > 0.5
            tmp1,tmp2,winner=play_gomoku(p1,p2,n=player_length(p1),n_in_row=min(5,(player_length(p1)+1)÷2),debug=debug)
        else
            tmp2,tmp1,winner=play_gomoku(p2,p1,n=player_length(p1),n_in_row=min(5,(player_length(p1)+1)÷2),debug=debug)
        end
        p1_data=cat(p1_data,deepcopy(tmp1),dims=1)
        p2_data=cat(p2_data,deepcopy(tmp2),dims=1)
    end
    return p1_data,p2_data
end

#my_crossentropy(x,y)=x.*log.(y)+(1 .-x).*log.(1 .-y) |> mean

function train_player(data,NN::player;batchsize=512,epoch=10)
    w,h,c=size(data[1][1])
    input=zeros(Float32,w,h,c,length(data))
    better_pi=zeros(w,h,1,length(data))
    real_v=zeros(1,1,1,length(data))

    for i=1:length(data)
        input[:,:,:,i]=data[i][1]
        better_pi[:,:,:,i]=data[i][2]
        real_v[1,1,1,i]=data[i][3]
    end

    function player_loss(x,better_pi,real_v)
        pi,v=NN(x)
        return sum(norm,params(NN))/3000+Flux.mse(v,real_v)-mean(better_pi.* log.(1e-10 .+pi))
    end

    if gpu_player(NN)
        input=cu(input)
        better_pi=cu(better_pi)
        real_v=cu(real_v)
    end

    batch_ind=zeros(Int64,min(batchsize,length(data)÷2))
    println("training...")
    for i=1:epoch
        StatsBase.self_avoid_sample!(1:length(data),batch_ind)
        Flux.train!(player_loss,params(NN),[(input[:,:,:,batch_ind],better_pi[:,:,:,batch_ind],real_v[1,1,1,batch_ind])],opt)
        if i%5==0
            println("epoch ",i," loss is ",player_loss(input,better_pi,real_v))
        end
    end
end

function play_match(p1,p2;n_games=30)
    wins=[0,0]
    for i =1:n_games
        _,_,winner=play_gomoku(p1,p2,debug=false)
        wins[winner]+=1
        _,_,winner=play_gomoku(p2,p1,debug=false)
        wins[3-winner]+=1
    end
    return wins
end

function self_play(p1;iterations=30,n_games=300,debug=false,test_games=30)
    #p2 is the data generator, p1 is the trained copy
    p2=deepcopy(p1)
    data=[]
    for iter=1:iterations
        println("iteration ",iter)
        tmp1,tmp2=gather_data(p2,p2,n_games=n_games,debug=debug)
        tmp3=cat(tmp1,tmp2,dims=1)
        data=cat(data,tmp3,dims=1)
        if length(data)>10000
            data=data[end-10000:end]
        end
        train_player(data,p1,epoch=200)

        wins=play_match(p1,p2,n_games=test_games)
        println(" trained p1 against best player p2 winrate ",wins[1]/sum(wins))
        if wins[1]/sum(wins)>0.55
            p2=deepcopy(p1)
            println("best player changed")
            p1_data=[]
            @save "best_5x5_player-$(now()).bson" p2
        end

    end
    return p1
end
opt=ADAM(0.01)

p1=player(5)|>gpu
@time p1(cu(rand(5,5,2,1)))
play_match(p1,p1,n_games=1)
p2=player(5)|>gpu
@time better_player=self_play(p2,iterations=1,n_games=1,debug=true,test_games=1)
@time better_player=self_play(p2,iterations=1,n_games=1,debug=true,test_games=1)

p1=self_play(p1,n_games=100,test_games=10)


p1=self_play(p1,iterations=500,n_games=20)

@save "test_net.bson" p1
p1_data,p2_data=gather_data(p1,p1,n_games=300,debug=true)
JLD2.@save "test_net_random_data.jld2" p1_data,p2_data

p2=deepcopy(p1)
data=cat(p1_data,p2_data,dims=1)
train_player(data,p2,epoch=40)
p3=deepcopy(p1)
data2=data[1:2:end]
train_player(data2,p3,epoch=200)

wins=play_match(p3,p1)

wins=[0,0]
for i=1:20
    _,_,winner1=play_gomoku(p1,p2,debug=false)
    wins[winner1]+=1
    _,_,winner1=play_gomoku(p2,p1,debug=false)
    wins[3-winner1]+=1
end
