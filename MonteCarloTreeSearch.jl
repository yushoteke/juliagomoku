include("juliagomoku.jl")

mutable struct MCTS_node
    G::gomoku
    N::Int32
    W::Float32
    Q::Float32
    V::Float32
    p::Float32
    is_leaf::Bool
    childs::Dict{CartesianIndex{2},MCTS_node}
end

#MCTS_node(G::gomoku)=MCTS_node(G,0,0,0,0,0,true,Dict{CartesianIndex{2},MCTS_node}())
MCTS_node(G::gomoku;p=0.0)=MCTS_node(G,0,0,0,0,p,true,Dict{CartesianIndex{2},MCTS_node}())

function print_node(node::MCTS_node)
    print_board(node.G)
    println("N:",node.N," W:",node.W," Q:",node.Q," V:",node.V," p:",node.p)
    println("Player:",node.G.player_playing," is_terminal:",node.G.is_terminal)
end

function print_childs(node::MCTS_node)
    for i in values(node.childs)
        print_node(i)
    end
end

function sample_child(node::MCTS_node,c=1.0)
    utilities = map(x->x.Q+c*x.p*sqrt(node.N/(1+x.N)),values(node.childs))
    val,ind=findmax(utilities)
    return node.childs[collect(keys(node.childs))[ind]]
end

function expand!(node::MCTS_node,NN)
    if node.G.is_terminal
        node.V = -1
    else
        w,h,c=size(node.G.B)
        pi,v=NN(reshape(current_perspective(node.G)+randn(Float32,w,h,c)/2048,w,h,c,1))
        node.V = typeof(v[1])==Tracker.TrackedReal{Float32} ? v[1].data : v[1]
        for i in valid_moves(node.G)
            tmp=deepcopy(node.G)
            play_move!(tmp,i)
            node.childs[i]=MCTS_node(tmp,p=typeof(v[1])==Tracker.TrackedReal{Float32} ? pi[i].data : pi[i])
        end
    end
end

function update_edge(node::MCTS_node,v::Float32)
    node.W += v
    node.N += 1
    node.Q = node.W / node.N
end

function simulation(node::MCTS_node,NN)
    if node.is_leaf
        expand!(node,NN)
        if !node.G.is_terminal
            node.is_leaf=false
        end
        v = node.V
    else
        v = simulation(sample_child(node),NN)
    end
    update_edge(node,v)
    return -v
end

function get_stronger_pi(G::gomoku,NN,sim_num::Int64=100,t=1.0)
    node = MCTS_node(G)
    for i=1:sim_num
        simulation(node,NN)
    end
    PI=zeros(Float32,5,5)
    if G.is_terminal
        return PI
    else
        for i in keys(node.childs)
            PI[i]=node.childs[i].N^t
        end
        return PI/sum(PI)
    end
end
